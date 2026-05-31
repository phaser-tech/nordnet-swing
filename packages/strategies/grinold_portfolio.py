"""Grinold-style portfolio of weak signals on OMX open->close (Rank 2 from
the edge-frontier memo, PR #23).

The idea: Grinold's fundamental law IR = IC * sqrt(breadth) says that the
information ratio of an active strategy scales with both per-signal skill (IC)
and the number of independent signals (breadth). All five Phase-0 OOS tests
used a single concentrated signal; this module tests whether the same signals
combined, frozen on train and applied blind to test, beat their best individual
component.

ARCHITECTURE
============
The strategy is a thin emitter that reads a *precomputed* prediction column
from the wide frame -- same single-source-of-truth pattern as ORB (#20) and
cross-asset gap (#22). The pipeline:

  1. Composition root (script) builds a wide frame with OMX OHLCV + each
     signal-source ticker's Close.
  2. `build_signal_columns` derives the per-signal shifted series (every
     signal uses `.shift(1)` so it only sees data <= T-1 close).
  3. `fit_ols` fits a linear regression of OMX open->close return on the
     signals using ONLY train rows. The fit returns frozen coefficients.
  4. `predict_series` applies those frozen coefficients to the full sample
     (train + test). For test rows this is genuinely out-of-sample.
  5. `attach_prediction` writes the predictions onto the wide frame as
     `GRINOLD_PREDICTION_COL`.
  6. `GrinoldPortfolioStrategy(threshold).generate_signals(wide_frame)`
     emits one Signal per row where `|prediction| > threshold`, direction
     = sign(prediction).

OOS DISCIPLINE
==============
- All signals are .shift(1)'d, so the value at index T uses only data <= T-1
  close. Decision at Stockholm open T uses the freshest available US-close
  T-1 data (which arrives at ~22:00 CET T-1, before Stockholm open T at
  09:00 CET T). This is the same lookahead discipline as `cross_asset_confluence`
  (PR #10), correctly applied to the open->close horizon.
- The OLS fit is performed only on rows with index < split_date (train).
- The threshold is computed only on train predictions (in the script).
- Test rows are never seen during fitting -- application is blind.

Stateless: instrument + signal specs + frozen coefficients + threshold are
configuration. Imports only `core` + pandas + numpy (dependency rule #6).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd

from packages.core.domain.signal import Conviction, Direction, Signal

GRINOLD_PREDICTION_COL = "_grinold_pred"

# The signal pool. Each entry maps a short name to (ticker, transform).
# Composition with: SPX trend captured via 2-day pct change (vs SMA which was
# used by cross_asset_confluence); raw vix_2d (the under-used strong signal
# from the EU/US analysis); plus the rest of the analysis-validated set.
SIGNAL_SPECS: Mapping[str, tuple[str, str]] = {
    # US signals (PR #10's source set, but raw -- no deadband, no z-scoring)
    "spx_2d": ("^GSPC", "pct2"),
    "vix_lvl": ("^VIX", "level"),
    "vix_2d": ("^VIX", "diff2"),
    "dxy_2d": ("DX-Y.NYB", "pct2"),
    "gold_2d": ("GC=F", "pct2"),
    "tnx_2d": ("^TNX", "diff2"),
    # EU signals added in the EU/US analysis (no individual significance there,
    # but Grinold's law says they still contribute via breadth if their
    # information is at all independent of the US set)
    "dax_2d": ("^GDAXI", "pct2"),
    "eursek_2d": ("EURSEK=X", "pct2"),
    "usdsek_2d": ("USDSEK=X", "pct2"),
    "wti_2d": ("CL=F", "pct2"),
    "copper_2d": ("HG=F", "pct2"),
}


@dataclass(frozen=True)
class FittedCoefficients:
    """Frozen linear-model coefficients from `fit_ols`. Treated as read-only
    once handed to the strategy."""

    signal_names: tuple[str, ...]
    intercept: float
    betas: tuple[float, ...]  # one per signal, ordered as signal_names

    def __post_init__(self) -> None:
        if len(self.betas) != len(self.signal_names):
            raise ValueError(
                f"got {len(self.betas)} betas for {len(self.signal_names)} signals"
            )


def _apply_transform(close: pd.Series, transform: str) -> pd.Series:
    if transform == "pct2":
        return close.pct_change(2)
    if transform == "diff2":
        return close.diff(2)
    if transform == "level":
        return close
    raise ValueError(f"unknown transform {transform!r}")


def build_signal_columns(
    wide_frame: pd.DataFrame,
    signal_specs: Mapping[str, tuple[str, str]] = SIGNAL_SPECS,
) -> pd.DataFrame:
    """Return a DataFrame of per-signal `.shift(1)` series, indexed like the
    wide frame. The wide frame must contain `{ticker}_Close` for every ticker
    referenced in `signal_specs`."""
    cols: dict[str, pd.Series] = {}
    for name, (ticker, transform) in signal_specs.items():
        col = f"{ticker}_Close"
        if col not in wide_frame.columns:
            raise ValueError(f"wide_frame is missing column {col!r} for signal {name!r}")
        raw = _apply_transform(wide_frame[col], transform)
        cols[name] = raw.shift(1)
    return pd.DataFrame(cols, index=wide_frame.index)


def fit_ols(
    signal_cols: pd.DataFrame,
    omx_open_close_return: pd.Series,
    train_mask: pd.Series,
) -> FittedCoefficients:
    """Fit OLS of OMX open->close return on the signal columns using ONLY rows
    where `train_mask` is True and no input is NaN. Returns frozen coefficients.

    Use the returned object verbatim for prediction on test data -- never refit
    on test, never let the strategy compute coefficients on the fly.
    """
    df = signal_cols.copy()
    df["__y"] = omx_open_close_return
    df = df.loc[train_mask].dropna()
    signal_names = tuple(signal_cols.columns)
    x = np.column_stack(
        [np.ones(len(df)), df[list(signal_names)].to_numpy(dtype=np.float64)]
    )
    y = df["__y"].to_numpy(dtype=np.float64)
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    return FittedCoefficients(
        signal_names=signal_names,
        intercept=float(coeffs[0]),
        betas=tuple(float(b) for b in coeffs[1:]),
    )


def predict_series(
    signal_cols: pd.DataFrame, coefficients: FittedCoefficients
) -> pd.Series:
    """Apply the frozen coefficients to every row of `signal_cols`. Rows with
    NaN in any signal get a NaN prediction (downstream the strategy skips them)."""
    arr = signal_cols[list(coefficients.signal_names)].to_numpy(dtype=np.float64)
    betas = np.array(coefficients.betas, dtype=np.float64)
    pred = coefficients.intercept + arr @ betas  # NaN propagates row-wise
    return pd.Series(pred, index=signal_cols.index, name=GRINOLD_PREDICTION_COL)


def attach_prediction(wide_frame: pd.DataFrame, prediction: pd.Series) -> pd.DataFrame:
    """Return a copy of the wide frame with the prediction column attached."""
    if prediction.name != GRINOLD_PREDICTION_COL:
        prediction = prediction.rename(GRINOLD_PREDICTION_COL)
    out = wide_frame.copy()
    out[GRINOLD_PREDICTION_COL] = prediction.reindex(wide_frame.index)
    return out


class GrinoldPortfolioStrategy:
    """Thin emitter: one Signal per row where `|_grinold_pred| > threshold`.

    The threshold is pre-registered on train and frozen for OOS. The strategy
    does NO fitting at runtime -- it just gates on the precomputed column.
    """

    def __init__(self, instrument: str, *, threshold: float) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        self._instrument = instrument
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "grinold_portfolio"

    def generate_signals(self, market_data: pd.DataFrame) -> Iterator[Signal]:
        if GRINOLD_PREDICTION_COL not in market_data.columns:
            raise ValueError(
                f"market_data must carry the precomputed {GRINOLD_PREDICTION_COL!r} "
                "column from `attach_prediction(...)`."
            )
        for ts, row in market_data.iterrows():
            pred = row[GRINOLD_PREDICTION_COL]
            if pred is None or (isinstance(pred, float) and np.isnan(pred)):
                continue
            pred_f = float(pred)
            if abs(pred_f) < self._threshold:
                continue
            direction = Direction.LONG if pred_f > 0 else Direction.SHORT
            entry = Decimal(str(float(row["Open"])))
            target_mult = Decimal("1.01") if direction == Direction.LONG else Decimal("0.99")
            stop_mult = Decimal("0.99") if direction == Direction.LONG else Decimal("1.01")
            yield Signal(
                timestamp=pd.Timestamp(ts).to_pydatetime(),
                strategy_name=self.name,
                instrument=self._instrument,
                direction=direction,
                conviction=Conviction.HIGH,
                suggested_entry=entry,
                suggested_stop=entry * stop_mult,
                suggested_target=entry * target_mult,
                confluence_factors=["grinold_aggregate"],
                notes=f"grinold {direction.value} (pred={pred_f * 100:+.4f}%)",
            )
