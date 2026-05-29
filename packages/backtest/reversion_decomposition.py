"""Decomposition of the next-day bounce after an extreme move (diagnostic).

The OOS harness answers "is there a tradeable net edge?" for a signal's bar
(open->close). This module answers a different, structural question: *where does
the reversion live* — in the overnight gap, the intraday session, or both?

For each trigger day T we split day T+1's full move into:
  - gap      : (open_T+1  - close_T)  / close_T
  - intraday : (close_T+1 - open_T+1) / open_T+1
  - full_day : (close_T+1 - close_T)  / close_T   (== (1+gap)(1+intraday) - 1)

Results are bucketed by the extreme's direction (down-extreme vs up-extreme),
because reversion has opposite sign in each. The caller supplies the per-day bet
direction (see `strategies.mean_reversion_extreme.bet_direction_series`) so this
module stays generic and never imports the strategies package (dependency rule
#6): the composition root wires the two together.

Why this matters: under the no-overnight hard rule the gap is NOT tradeable
(capturing it means holding close_T -> open_T+1). If the reversion sits in the
gap while the intraday leg is flat, the "edge" is an untradeable overnight
artifact — the gap-arbitrage pattern. If the intraday leg carries the reversion,
there is a real candidate to cost-test.

All math is `float` (analytics layer, not the trading domain).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import pandas as pd

from packages.core.domain.signal import Direction

GAP = "gap"
INTRADAY = "intraday"
FULL_DAY = "full_day"
LEG_COLUMNS = (GAP, INTRADAY, FULL_DAY)


@dataclass(frozen=True)
class LegStats:
    """Stats for one leg (gap / intraday / full-day) within one direction bucket."""

    n: int
    mean: float  # raw mean underlying return for this leg (sign as observed)
    t_stat: float  # significance of the raw mean vs zero
    reversion_mean: float  # raw mean signed by the bet (+ve => move was reverting)
    reversion_hit_rate: float  # fraction of events that moved in the reversion direction


@dataclass(frozen=True)
class BucketDecomposition:
    """All three legs for one extreme-direction bucket."""

    name: str  # "down-extreme" or "up-extreme"
    bet_direction: Direction  # LONG after a down-extreme, SHORT after an up-extreme
    n_events: int
    gap: LegStats
    intraday: LegStats
    full_day: LegStats


def next_day_legs(bars: pd.DataFrame) -> pd.DataFrame:
    """Per trigger day T, the three legs of day T+1's move. Indexed by T.

    The last bar has no T+1 and is dropped. Identity holds per row:
    `(1 + gap) * (1 + intraday) == 1 + full_day`.
    """
    close = bars["Close"]
    open_next = bars["Open"].shift(-1)
    close_next = close.shift(-1)
    legs = pd.DataFrame(
        {
            GAP: open_next / close - 1.0,
            INTRADAY: close_next / open_next - 1.0,
            FULL_DAY: close_next / close - 1.0,
        },
        index=bars.index,
    )
    return legs.dropna()


def _leg_stats(values: list[float], bet_sign: float) -> LegStats:
    """Reduce one leg's per-event returns to summary stats, given the bet sign."""
    n = len(values)
    if n == 0:
        return LegStats(n=0, mean=0.0, t_stat=0.0, reversion_mean=0.0, reversion_hit_rate=0.0)
    mean = statistics.fmean(values)
    if n > 1 and statistics.stdev(values) > 0.0:
        t_stat = mean / (statistics.stdev(values) / math.sqrt(n))
    else:
        t_stat = 0.0
    hits = sum(1 for v in values if v * bet_sign > 0.0)
    return LegStats(
        n=n,
        mean=mean,
        t_stat=t_stat,
        reversion_mean=mean * bet_sign,
        reversion_hit_rate=hits / n,
    )


def decompose(bars: pd.DataFrame, bet: pd.Series) -> dict[str, BucketDecomposition]:
    """Bucket the next-day legs by extreme direction.

    Args:
        bars: OHLCV frame (the trigger-day calendar).
        bet: per-day `Direction | None` from `bet_direction_series`, indexed like
             `bars`. LONG marks a down-extreme day, SHORT an up-extreme day.

    Returns a dict keyed "down-extreme" / "up-extreme".
    """
    legs = next_day_legs(bars)
    bet_aligned = bet.reindex(legs.index)

    buckets: dict[str, BucketDecomposition] = {}
    for name, direction in (("down-extreme", Direction.LONG), ("up-extreme", Direction.SHORT)):
        mask = bet_aligned.map(lambda d, _dir=direction: d is _dir)
        sub = legs[mask.astype(bool)]
        bet_sign = 1.0 if direction == Direction.LONG else -1.0
        buckets[name] = BucketDecomposition(
            name=name,
            bet_direction=direction,
            n_events=len(sub),
            gap=_leg_stats([float(v) for v in sub[GAP]], bet_sign),
            intraday=_leg_stats([float(v) for v in sub[INTRADAY]], bet_sign),
            full_day=_leg_stats([float(v) for v in sub[FULL_DAY]], bet_sign),
        )
    return buckets
