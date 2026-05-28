"""Unit tests for the volume-confirmed breakout strategy. Fixtures only."""

from __future__ import annotations

import pandas as pd
import pytest

from packages.core.domain.signal import Direction
from packages.strategies.breakout_confluence import BreakoutConfluenceStrategy

# Tiny windows -> warmup = max(3,3,3)+2 = 5.
_PARAMS = dict(level_window=3, volume_window=3, volume_multiple=1.5, expected_move_window=3)


def _frame(close: list[float], volume: list[float]) -> pd.DataFrame:
    n = len(close)
    idx = pd.DatetimeIndex(
        pd.to_datetime([f"2024-01-{i + 1:02d}" for i in range(n)]), name="ts"
    ).tz_localize("UTC")
    return pd.DataFrame(
        {
            "Open": [c - 0.2 for c in close],
            "High": [c + 0.5 for c in close],
            "Low": [c - 0.5 for c in close],
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )


# Flat ~100 (establishes the prior-3d level), then an up-break at index 4 on
# 2x volume -> LONG trade at index 5.
_UP_CLOSE = [100.0, 100.0, 100.0, 100.0, 105.0, 104.0, 103.0, 102.0]
_UP_VOL = [1000.0, 1000.0, 1000.0, 1000.0, 2000.0, 1000.0, 1000.0, 1000.0]


class TestConstruction:
    def test_rejects_bad_params(self) -> None:
        with pytest.raises(ValueError, match="level_window"):
            BreakoutConfluenceStrategy("^OMX", level_window=0)
        with pytest.raises(ValueError, match="volume_multiple"):
            BreakoutConfluenceStrategy("^OMX", volume_multiple=0)


class TestBreakoutDetection:
    def test_up_break_on_volume_goes_long_next_day(self) -> None:
        strat = BreakoutConfluenceStrategy("^OMX", **_PARAMS)
        signals = list(strat.generate_signals(_frame(_UP_CLOSE, _UP_VOL)))
        assert len(signals) == 1
        s = signals[0]
        assert s.direction == Direction.LONG
        assert s.timestamp.day == 6  # index 5 -> 2024-01-06 (break was index 4)
        assert "breakout_above_20d_high" in s.confluence_factors

    def test_down_break_on_volume_goes_short_next_day(self) -> None:
        close = [100.0, 100.0, 100.0, 100.0, 95.0, 96.0, 97.0, 98.0]
        strat = BreakoutConfluenceStrategy("^OMX", **_PARAMS)
        signals = list(strat.generate_signals(_frame(close, _UP_VOL)))
        assert len(signals) == 1
        assert signals[0].direction == Direction.SHORT
        assert "breakout_below_20d_low" in signals[0].confluence_factors

    def test_break_without_volume_is_ignored(self) -> None:
        low_vol = [1000.0] * 8  # no volume spike on the break day
        strat = BreakoutConfluenceStrategy("^OMX", **_PARAMS)
        assert list(strat.generate_signals(_frame(_UP_CLOSE, low_vol))) == []

    def test_no_break_is_flat(self) -> None:
        flat = [100.0, 100.2, 99.8, 100.1, 99.9, 100.0, 100.1, 99.9]
        strat = BreakoutConfluenceStrategy("^OMX", **_PARAMS)
        assert list(strat.generate_signals(_frame(flat, _UP_VOL))) == []


class TestNoLookahead:
    def test_day_t_decision_ignores_day_t_data(self) -> None:
        strat = BreakoutConfluenceStrategy("^OMX", **_PARAMS)
        frame = _frame(_UP_CLOSE, _UP_VOL)
        before = {s.timestamp: s for s in strat.generate_signals(frame)}
        assert before
        day_d = next(iter(before))  # the LONG trade day (index 5)
        pos = frame.index.get_loc(pd.Timestamp(day_d))

        # Poison the TRADE day's own OHLCV: must not change that day's signal.
        poisoned = frame.copy()
        for col in ("Open", "High", "Low", "Close", "Volume"):
            poisoned.iloc[pos, poisoned.columns.get_loc(col)] = 1.0
        after = {s.timestamp: s for s in strat.generate_signals(poisoned)}
        assert after.get(day_d) == before[day_d]

    def test_signal_depends_on_breakout_day(self) -> None:
        # Poisoning the BREAKOUT day (T-1) must remove the next-day signal.
        strat = BreakoutConfluenceStrategy("^OMX", **_PARAMS)
        frame = _frame(_UP_CLOSE, _UP_VOL)
        before = {s.timestamp: s for s in strat.generate_signals(frame)}
        day_d = next(iter(before))
        pos = frame.index.get_loc(pd.Timestamp(day_d))

        poisoned = frame.copy()
        # Kill the break: set breakout-day (pos-1) close back into the range.
        poisoned.iloc[pos - 1, poisoned.columns.get_loc("Close")] = 100.0
        after = {s.timestamp: s for s in strat.generate_signals(poisoned)}
        assert day_d not in after
