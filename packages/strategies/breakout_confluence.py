"""Technical confluence: volume-confirmed breakout of a multi-day key level.

Setup (frozen a priori — issue #11): break-and-continue of a 20-day Donchian
level on elevated volume.
  - LONG  when close breaks ABOVE the prior 20-day high on volume >= 1.5x the
    trailing 20-day average.
  - SHORT when close breaks BELOW the prior 20-day low on the same volume gate.

We do NOT use mean-reversion / test-and-fade: CLAUDE.md excludes "pure mean
reversion without macro filter". Breakout/continuation also targets larger
moves, which is what's needed to clear the 5x cost hurdle.

TIMING / LOOKAHEAD: the breakout is confirmed at the day T-1 close (we need the
close to know the level broke and to read the day's volume). Earliest tradeable
entry is therefore day T's open; we trade T intraday open->close (runner's
no-overnight model). The level is the prior-20-day high/low EXCLUDING the
breakout day. No day-T data enters the day-T decision (see the no-lookahead
test). Stateless; imports only core + pandas.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal

import pandas as pd

from packages.core.domain.signal import Conviction, Direction, Signal


class BreakoutConfluenceStrategy:
    """Volume-confirmed 20-day breakout, long on up-break / short on down-break."""

    def __init__(
        self,
        instrument: str,
        *,
        level_window: int = 20,
        volume_window: int = 20,
        volume_multiple: float = 1.5,
        expected_move_window: int = 20,
    ) -> None:
        for n, v in (
            ("level_window", level_window),
            ("volume_window", volume_window),
            ("expected_move_window", expected_move_window),
        ):
            if v <= 0:
                raise ValueError(f"{n} must be > 0, got {v}")
        if volume_multiple <= 0:
            raise ValueError(f"volume_multiple must be > 0, got {volume_multiple}")
        self._instrument = instrument
        self._level_window = level_window
        self._volume_window = volume_window
        self._volume_multiple = volume_multiple
        self._expected_move_window = expected_move_window

    @property
    def name(self) -> str:
        return f"breakout_confluence_{self._level_window}"

    def generate_signals(self, market_data: pd.DataFrame) -> Iterator[Signal]:
        df = market_data
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        volume = df["Volume"]

        # Prior N-day level, EXCLUDING the current day (shift(1)).
        upper = high.rolling(self._level_window).max().shift(1)
        lower = low.rolling(self._level_window).min().shift(1)
        avg_vol = volume.rolling(self._volume_window).mean().shift(1)
        vol_ok = volume >= self._volume_multiple * avg_vol

        broke_up = (close > upper) & vol_ok
        broke_down = (close < lower) & vol_ok
        # The break happens on T-1; we trade the next day -> shift the trade flag.
        trade_long = broke_up.shift(1).fillna(False)
        trade_short = broke_down.shift(1).fillna(False)

        close_prev = close.shift(1)  # advisory entry reference (T-1), no lookahead
        move = (
            close.pct_change().abs().rolling(self._expected_move_window).mean().shift(1)
        )

        warmup = (
            max(self._level_window, self._volume_window, self._expected_move_window) + 2
        )

        for i, ts in enumerate(df.index):
            if i < warmup:
                continue
            go_long = bool(trade_long.loc[ts])
            go_short = bool(trade_short.loc[ts])
            if not (go_long or go_short):
                continue

            m = move.loc[ts]
            ref = close_prev.loc[ts]
            if pd.isna(m) or pd.isna(ref) or m <= 0:
                continue

            direction = Direction.LONG if go_long else Direction.SHORT
            entry = Decimal(str(ref))
            move_dec = Decimal(str(m))
            if direction == Direction.LONG:
                target = entry * (Decimal("1") + move_dec)
                stop = entry * (Decimal("1") - move_dec)
                factors = ["breakout_above_20d_high", "volume_confirmed"]
            else:
                target = entry * (Decimal("1") - move_dec)
                stop = entry * (Decimal("1") + move_dec)
                factors = ["breakout_below_20d_low", "volume_confirmed"]

            yield Signal(
                timestamp=ts.to_pydatetime(),
                strategy_name=self.name,
                instrument=self._instrument,
                direction=direction,
                conviction=Conviction.HIGH,
                suggested_entry=entry,
                suggested_stop=stop,
                suggested_target=target,
                confluence_factors=factors,
                notes=f"{self.name}: volume-confirmed {direction.value} breakout",
            )
