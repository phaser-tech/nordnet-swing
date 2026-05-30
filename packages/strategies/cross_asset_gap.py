"""Cross-asset gap-capture strategy (issue #21).

Same risk-on/off signal as `cross_asset_confluence` (PR #10) -- read this
module as "PR #10's signal source, on a different trade horizon". The
hypothesis under test: four open->close OOS failures (#10, #14, #15, #20)
diagnostically pointed at the overnight gap as the dominant directional
mover. If that diagnosis is right, the *same* cross-asset signal should
produce net edge on the gap leg `close(T-1) -> open(T)` -- exactly the leg
the open->close trades cannot capture under the no-overnight rule.

We trade ONE night per signal. This is allowed via the narrow exception
ratified in #21 and registered in CLAUDE.md "Approved overnight exceptions".
The cost model includes the one-night financing term via
`estimate_round_trip_cost(overnight_nights=1)`.

Architectural seam (same pattern as ORB in #20): the strategy operates on
the cross-asset wide frame but `oos.py` scores `sign * (close - open) / open`
per signal bar. We build a synthetic per-day frame whose Open is the prior
day's OMX close and Close is the current day's OMX open -- so `oos.py`'s
formula evaluates the signed gap return. The signal source (which days fire,
in which direction) is unchanged from PR #10, so any difference vs PR #10's
result is attributable to the horizon change, not the signal.

Stateless: only the instrument and the optional column names are config.
Imports only `core` and pandas (dependency rule #6).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from decimal import Decimal

import pandas as pd

from packages.core.domain.signal import Conviction, Direction, Signal

GAP_DIRECTION_COL = "_gap_direction"
_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_OUTPUT_COLUMNS = (*_OHLCV_COLUMNS, GAP_DIRECTION_COL)


def build_per_day_gap(
    wide_frame: pd.DataFrame,
    signals: Iterable[Signal],
    *,
    underlying_close_col: str = "Close",
    underlying_open_col: str = "Open",
) -> pd.DataFrame:
    """Collapse the cross-asset wide frame into a per-day frame that, fed to
    `oos.py`, scores the *gap* return `sign * (open_T - close_{T-1}) / close_{T-1}`.

    For each trading day T (skipping the first row, where T-1 is undefined):
      - if a signal fires on T: Open = close at T-1, Close = open at T,
        `_gap_direction` = signal direction.
      - else: Open = Close = close at T-1 (placeholder; never read by oos.py
        since there's no signal there) and `_gap_direction` = None.

    The synthetic frame's index equals `wide_frame.index[1:]` so signal
    timestamps align directly. The signals are generic -- the caller provides
    any iterable of `Signal` objects, decoupling the build from any specific
    cross-asset strategy implementation (rule #6: strategies don't import each
    other through indirect channels; the composition root passes them in).
    """
    if wide_frame.empty:
        return _empty_per_day_frame()
    if len(wide_frame) < 2:
        return _empty_per_day_frame()

    signals_by_ts = {pd.Timestamp(s.timestamp): s for s in signals}

    closes = wide_frame[underlying_close_col]
    opens = wide_frame[underlying_open_col]

    rows: list[dict[str, object]] = []
    for i in range(1, len(wide_frame)):
        ts = wide_frame.index[i]
        close_prev = float(closes.iloc[i - 1])
        open_t = float(opens.iloc[i])
        sig = signals_by_ts.get(pd.Timestamp(ts))
        if sig is None:
            rows.append(
                {
                    "ts": ts,
                    "Open": close_prev,
                    "High": close_prev,
                    "Low": close_prev,
                    "Close": close_prev,
                    "Volume": 0,
                    GAP_DIRECTION_COL: None,
                }
            )
        else:
            rows.append(
                {
                    "ts": ts,
                    "Open": close_prev,  # entry leg = prior day's close
                    "High": max(close_prev, open_t),  # placeholder; unused
                    "Low": min(close_prev, open_t),
                    "Close": open_t,  # exit leg = current day's open
                    "Volume": 0,
                    GAP_DIRECTION_COL: sig.direction,
                }
            )

    frame = pd.DataFrame(rows).set_index("ts")
    frame.index = pd.DatetimeIndex(frame.index, name="ts")
    return frame[list(_OUTPUT_COLUMNS)]


def _empty_per_day_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(_OUTPUT_COLUMNS))
    frame.index = pd.DatetimeIndex([], name="ts")
    return frame


class CrossAssetGapStrategy:
    """Thin emitter over the synthetic per-day gap frame -- one Signal per row
    where `_gap_direction` is set. Matches the structural pattern of #20's ORB.
    """

    def __init__(self, instrument: str) -> None:
        self._instrument = instrument

    @property
    def name(self) -> str:
        return "cross_asset_gap"

    def generate_signals(self, market_data: pd.DataFrame) -> Iterator[Signal]:
        if GAP_DIRECTION_COL not in market_data.columns:
            raise ValueError(
                f"market_data must be the synthetic per-day frame produced by "
                f"build_per_day_gap (missing column {GAP_DIRECTION_COL!r})."
            )
        for ts, row in market_data.iterrows():
            direction = row[GAP_DIRECTION_COL]
            if not isinstance(direction, Direction):
                continue
            entry = Decimal(str(row["Open"]))
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
                confluence_factors=["cross_asset_gap"],
                notes=f"cross-asset gap-capture {direction.value} (close T-1 -> open T)",
            )
