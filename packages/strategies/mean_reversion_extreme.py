"""Mean-reversion-after-extreme-move strategy (Tier-1 edge source #3).

Thesis: after OMX has an unusually large daily move on day T (>= `sigma`
trailing standard deviations), day T+1 tends to revert. We bet the reversion
intraday on T+1: a down-extreme on T -> LONG on T+1 (bet on a bounce up); an
up-extreme on T -> SHORT on T+1 (bet on a fade down).

Selective by construction: only the tails of the daily-return distribution
trigger, so the strategy is flat the large majority of days (CLAUDE.md: default
state is no-trade).

LOOKAHEAD SAFETY (load-bearing): the extremeness of day T is measured from
day-T's close against a *trailing* return distribution that EXCLUDES day T (the
rolling stats are `.shift(1)`), so a single outlier never shrinks its own
z-score. Every decision input is therefore known by day T's close, i.e. before
day T+1's open — and the signal is emitted timestamped on the T+1 bar, which is
the bar the runner/OOS harness fills open->close. Day T+1's own prices never
enter the day-T+1 decision. See the no-lookahead test.

The signal carries the T+1 trade. Under the no-overnight hard rule we cannot
hold the close_T -> open_T+1 gap; only the open->close intraday leg of T+1 is
tradeable, which is exactly what `oos.py` evaluates for a signal on that bar.

Stateless: instrument, window, and sigma are configuration. Imports only `core`
+ pandas (no `market_data`, no sibling packages) per dependency rule #6.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from decimal import Decimal

import pandas as pd

from packages.core.domain.signal import Conviction, Direction, Signal

CLOSE = "Close"


def extreme_zscores(close: pd.Series, window: int) -> pd.Series:
    """Trailing z-score of daily returns, with stats that EXCLUDE the day itself.

    `z_T = (r_T - mean(r over [T-window, T-1])) / std(r over [T-window, T-1])`,
    where `r = close.pct_change()`. The `.shift(1)` on the rolling stats keeps the
    current return out of its own mean/std so a large move is scored against the
    regime that preceded it. Uses only data <= close_T. Degenerate (zero-variance
    or warmup) windows yield NaN, treated downstream as "no signal".
    """
    r = close.pct_change()
    mu = r.rolling(window).mean().shift(1)
    sd = r.rolling(window).std(ddof=0).shift(1)
    z = (r - mu) / sd
    return z.replace([float("inf"), float("-inf")], float("nan"))


def _bet(z_value: float, sigma: float) -> Direction | None:
    """Reversion bet for a day's z-score: LONG below -sigma, SHORT above +sigma."""
    if math.isnan(z_value):
        return None
    if z_value <= -sigma:
        return Direction.LONG  # down-extreme -> bet on the bounce up
    if z_value >= sigma:
        return Direction.SHORT  # up-extreme -> bet on the fade down
    return None


def bet_direction_series(bars: pd.DataFrame, *, window: int, sigma: float) -> pd.Series:
    """Per day T, the reversion bet direction if T is a >=sigma extreme, else None.

    Single source of truth for "which days trigger and in which direction" —
    `generate_signals` (the trades) and the decomposition diagnostic both derive
    from this, so the diagnostic describes exactly the trades the verdict scores.
    Indexed identically to `bars`.
    """
    z = extreme_zscores(bars[CLOSE], window)
    return z.map(lambda v: _bet(float(v), sigma))


class MeanReversionExtremeStrategy:
    """Bets reversion on T+1 after a >=sigma daily move on T. Satisfies StrategyLike."""

    def __init__(self, instrument: str, *, window: int = 30, sigma: float = 2.0) -> None:
        if window <= 1:
            raise ValueError(f"window must be > 1, got {window}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self._instrument = instrument
        self._window = window
        self._sigma = sigma

    @property
    def name(self) -> str:
        return "mean_reversion_extreme"

    def generate_signals(self, market_data: pd.DataFrame) -> Iterator[Signal]:
        df = market_data
        close = df[CLOSE]
        z = extreme_zscores(close, self._window)
        # Trailing daily vol (same shift discipline) — advisory target/stop scale.
        sd = close.pct_change().rolling(self._window).std(ddof=0).shift(1)
        index = df.index

        for i in range(len(index) - 1):  # need a T+1 bar to trade
            direction = _bet(float(z.iloc[i]), self._sigma)
            if direction is None:
                continue
            close_t = close.iloc[i]
            scale = sd.iloc[i]
            if pd.isna(close_t) or pd.isna(scale) or scale <= 0:
                continue

            entry = Decimal(str(close_t))  # advisory ref (oos fills the T+1 open)
            move = Decimal(str(scale))
            if direction == Direction.LONG:
                target, stop = entry * (1 + move), entry * (1 - move)
            else:
                target, stop = entry * (1 - move), entry * (1 + move)

            side = "down" if direction == Direction.LONG else "up"
            yield Signal(
                timestamp=index[i + 1].to_pydatetime(),  # trade day T+1
                strategy_name=self.name,
                instrument=self._instrument,
                direction=direction,
                conviction=Conviction.HIGH,
                suggested_entry=entry,
                suggested_stop=stop,
                suggested_target=target,
                confluence_factors=["extreme_daily_move"],
                notes=f"T z={float(z.iloc[i]):.2f} ({side}-extreme); revert {direction.value} on T+1",
            )
