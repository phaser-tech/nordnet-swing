"""Cross-asset confluence strategy — our first candidate-edge strategy.

A daily regime gate: it decides whether to take a same-day intraday trade on the
traded instrument (OMX) based on whether multiple cross-asset signals agree on
risk direction. Selective by design — fires only at high agreement.

Five signals, each scored -1 (risk-off) / 0 (neutral) / +1 (risk-on):
  - SPX trend   : close vs N-day SMA (the persistent +/-1 anchor)
  - VIX         : level + 2-day change, z-scored, sign-flipped (high/rising = off)
  - 10y yield   : 2-day change, z-scored (rising = risk-on)
  - DXY (USD)   : 2-day change, z-scored (rising USD = risk-off)
  - Gold        : 2-day change, z-scored (rising gold = risk-off)

Confluence: LONG when >= `min_agree` are risk-on, SHORT when >= `min_agree` are
risk-off, else flat.

Scoring is **adaptive**: change/level signals are normalized by their own trailing
volatility (z-score over `zscore_window`) and gated by a deadband `|z| < k`, so a
"meaningful move" auto-adapts per asset and per regime — no era-specific magic
numbers. A steady drift produces z~0 (not a signal); a *surprise* move triggers.
Selectivity emerges from the deadband AND the >=4/5 rule compounding.

LOOKAHEAD SAFETY (load-bearing): Stockholm opens (09:00 CET) before the US trades
(15:30 CET), so at Stockholm day T's open the most recent US close is day T-1.
Every decision input is `.shift(1)` (one trading day) before scoring, and all
trailing stats are computed on the shifted series. Day-T cross-asset/OMX values
never enter the day-T decision — see the no-lookahead test. The runner supplies
the actual day-T Open/Close fills; the strategy never uses them to decide.

Stateless: instrument, windows, and thresholds are configuration. Imports only
`core` + pandas (no `market_data`); the wide input frame is assembled by the
composition root.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from decimal import Decimal

import pandas as pd

from packages.core.domain.signal import Conviction, Direction, Signal

# Wide-frame column contract (assembled by the composition root).
OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
SPX_CLOSE = "SPX_Close"
VIX_CLOSE = "VIX_Close"
TNX_CLOSE = "TNX_Close"
DXY_CLOSE = "DXY_Close"
GOLD_CLOSE = "GOLD_Close"
CROSS_ASSET_COLUMNS = (SPX_CLOSE, VIX_CLOSE, TNX_CLOSE, DXY_CLOSE, GOLD_CLOSE)

# Stable names for the five scores, in order (used for confluence_factors).
SIGNAL_NAMES = ("spx_trend", "vix", "tnx_10y", "dxy_usd", "gold")


def _threshold(value: float, k: float) -> int:
    """Map a (signed, risk-on-oriented) value to -1/0/+1 with a deadband |.|<k."""
    if pd.isna(value):
        return 0
    if value > k:
        return 1
    if value < -k:
        return -1
    return 0


def _trend_score(value: float, sma: float) -> int:
    """+1 above SMA, -1 below, 0 if undefined (warmup)."""
    if pd.isna(value) or pd.isna(sma):
        return 0
    if value > sma:
        return 1
    if value < sma:
        return -1
    return 0


def _confluence(scores: Sequence[int], min_agree: int) -> tuple[Direction | None, int]:
    """Apply the >= min_agree rule. Returns (direction or None, winning count)."""
    n_on = sum(1 for s in scores if s == 1)
    n_off = sum(1 for s in scores if s == -1)
    if n_on >= min_agree:
        return Direction.LONG, n_on
    if n_off >= min_agree:
        return Direction.SHORT, n_off
    return None, max(n_on, n_off)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Trailing z-score. Zero-variance windows -> NaN (treated as neutral)."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    z = (series - mean) / std
    return z.replace([float("inf"), float("-inf")], float("nan"))


class CrossAssetConfluenceStrategy:
    """Risk-on/off cross-asset gate. Emits LONG/SHORT only at high agreement."""

    def __init__(
        self,
        instrument: str,
        *,
        spx_sma_window: int = 50,
        zscore_window: int = 60,
        deadband_k: float = 0.5,
        min_agree: int = 4,
        change_lookback: int = 2,
        expected_move_window: int = 20,
    ) -> None:
        if not 1 <= min_agree <= 5:
            raise ValueError(f"min_agree must be in 1..5, got {min_agree}")
        for n, v in (
            ("spx_sma_window", spx_sma_window),
            ("zscore_window", zscore_window),
            ("change_lookback", change_lookback),
            ("expected_move_window", expected_move_window),
        ):
            if v <= 0:
                raise ValueError(f"{n} must be > 0, got {v}")
        self._instrument = instrument
        self._spx_sma_window = spx_sma_window
        self._zscore_window = zscore_window
        self._deadband_k = deadband_k
        self._min_agree = min_agree
        self._change_lookback = change_lookback
        self._expected_move_window = expected_move_window

    @property
    def name(self) -> str:
        return "cross_asset_confluence"

    def generate_signals(self, market_data: pd.DataFrame) -> Iterator[Signal]:
        df = market_data
        k = self._deadband_k
        lb = self._change_lookback

        # All decision inputs shifted one trading day: day-T decision uses <= T-1.
        spx = df[SPX_CLOSE].shift(1)
        vix = df[VIX_CLOSE].shift(1)
        tnx = df[TNX_CLOSE].shift(1)
        dxy = df[DXY_CLOSE].shift(1)
        gold = df[GOLD_CLOSE].shift(1)

        spx_sma = spx.rolling(self._spx_sma_window).mean()
        spx_score = pd.Series(
            [_trend_score(c, m) for c, m in zip(spx, spx_sma, strict=True)],
            index=df.index,
        )

        # VIX: high level and/or rising change => risk-off, so flip the sign.
        vix_val = -(
            _zscore(vix, self._zscore_window)
            + _zscore(vix.diff(lb), self._zscore_window)
        )
        tnx_val = _zscore(tnx.diff(lb), self._zscore_window)  # rising = risk-on
        dxy_val = -_zscore(dxy.diff(lb), self._zscore_window)  # rising USD = risk-off
        gold_val = -_zscore(gold.diff(lb), self._zscore_window)  # rising gold = risk-off

        vix_score = vix_val.apply(lambda v: _threshold(v, k))
        tnx_score = tnx_val.apply(lambda v: _threshold(v, k))
        dxy_score = dxy_val.apply(lambda v: _threshold(v, k))
        gold_score = gold_val.apply(lambda v: _threshold(v, k))

        # Expected-move proxy (OMX realized daily move as of T-1) for the cost filter,
        # plus an advisory entry reference. Both use only <= T-1 data.
        omx_close_prev = df["Close"].shift(1)
        move = (
            df["Close"].pct_change().abs().rolling(self._expected_move_window).mean().shift(1)
        )

        warmup = max(self._spx_sma_window, self._zscore_window + lb)

        for i, ts in enumerate(df.index):
            if i < warmup:
                continue
            scores = [
                int(spx_score.loc[ts]),
                int(vix_score.loc[ts]),
                int(tnx_score.loc[ts]),
                int(dxy_score.loc[ts]),
                int(gold_score.loc[ts]),
            ]
            direction, agree = _confluence(scores, self._min_agree)
            if direction is None:
                continue

            m = move.loc[ts]
            ref = omx_close_prev.loc[ts]
            if pd.isna(m) or pd.isna(ref) or m <= 0:
                continue

            entry = Decimal(str(ref))
            move_dec = Decimal(str(m))
            if direction == Direction.LONG:
                target = entry * (Decimal("1") + move_dec)
                stop = entry * (Decimal("1") - move_dec)
            else:
                target = entry * (Decimal("1") - move_dec)
                stop = entry * (Decimal("1") + move_dec)

            want = 1 if direction == Direction.LONG else -1
            factors = [SIGNAL_NAMES[j] for j, s in enumerate(scores) if s == want]
            yield Signal(
                timestamp=ts.to_pydatetime(),
                strategy_name=self.name,
                instrument=self._instrument,
                direction=direction,
                conviction=Conviction.VERY_HIGH if agree == 5 else Conviction.HIGH,
                suggested_entry=entry,
                suggested_stop=stop,
                suggested_target=target,
                confluence_factors=factors,
                notes=f"{agree}/5 cross-asset signals agree ({direction.value})",
            )
