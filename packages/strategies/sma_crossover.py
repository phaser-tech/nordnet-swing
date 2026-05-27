"""SMA crossover strategy.

NOT A REAL EDGE STRATEGY. This exists purely to validate the backtest framework
end to end (strategy -> signal -> cost filter -> sizing -> simulation -> report).
It has no demonstrated edge and must not be traded. Real strategies require a
backtest report proving out-of-sample EV after costs (CLAUDE.md).

Logic: long when the fast SMA is above the slow SMA, flat otherwise. The
crossover state is evaluated as of the *prior* close (shifted one bar), so the
signal for bar t uses no information from bar t itself — the runner then fills
the trade intraday (open->close) on bar t. Stateless: instrument and window
lengths are configuration, not mutable state.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal

import pandas as pd

from packages.core.domain.signal import Conviction, Direction, Signal


class SMACrossoverStrategy:
    """Fast/slow SMA crossover, long/flat. Plumbing only — no real edge."""

    def __init__(
        self,
        instrument: str,
        fast: int = 10,
        slow: int = 30,
        expected_move_lookback: int = 20,
    ) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self._instrument = instrument
        self._fast = fast
        self._slow = slow
        self._expected_move_lookback = expected_move_lookback

    @property
    def name(self) -> str:
        return f"sma_crossover_{self._fast}_{self._slow}"

    def generate_signals(self, market_data: pd.DataFrame) -> Iterator[Signal]:
        closes = market_data["Close"]

        fast_sma = closes.rolling(self._fast).mean()
        slow_sma = closes.rolling(self._slow).mean()
        # Crossover state as of the prior close -> act on the current bar.
        long_today = (fast_sma > slow_sma).shift(1)

        # Expected intraday move proxy: recent mean absolute daily return,
        # shifted so it uses only prior-bar information. Crude on purpose.
        exp_move = (
            closes.pct_change()
            .abs()
            .rolling(self._expected_move_lookback)
            .mean()
            .shift(1)
        )
        prior_close = closes.shift(1)  # advisory entry reference (no lookahead)

        for ts in closes.index:
            if not bool(long_today.loc[ts]):
                continue
            move = exp_move.loc[ts]
            ref = prior_close.loc[ts]
            if pd.isna(move) or pd.isna(ref) or move <= 0:
                continue  # warmup / no usable expected move

            entry = Decimal(str(ref))
            move_dec = Decimal(str(move))
            yield Signal(
                timestamp=ts.to_pydatetime(),
                strategy_name=self.name,
                instrument=self._instrument,
                direction=Direction.LONG,
                conviction=Conviction.MEDIUM,
                suggested_entry=entry,
                suggested_stop=entry * (Decimal("1") - move_dec),
                suggested_target=entry * (Decimal("1") + move_dec),
                confluence_factors=["fast_sma_above_slow_sma"],
                notes=f"{self.name}: fast>slow as of prior close",
            )
