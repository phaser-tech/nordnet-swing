"""Unit tests for the SMA crossover plumbing strategy."""

from __future__ import annotations

import pandas as pd
import pytest

from packages.core.domain.signal import Direction
from packages.strategies.sma_crossover import SMACrossoverStrategy


def _bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        pd.to_datetime([f"2024-01-{i + 1:02d}" for i in range(len(closes))]),
        name="ts",
    ).tz_localize("UTC")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": [1000] * len(closes),
        },
        index=idx,
    )


class TestConstruction:
    def test_fast_must_be_below_slow(self) -> None:
        with pytest.raises(ValueError, match="fast"):
            SMACrossoverStrategy("TEST", fast=30, slow=10)


class TestSignalGeneration:
    def test_monotonic_uptrend_emits_long_signals(self) -> None:
        bars = _bars([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        strat = SMACrossoverStrategy("TEST", fast=2, slow=3, expected_move_lookback=2)
        signals = list(strat.generate_signals(bars))

        # fast>slow holds from idx2; shifted one bar -> signals from idx3 onward.
        assert [s.timestamp for s in signals] == list(bars.index[3:].to_pydatetime())
        assert all(s.direction == Direction.LONG for s in signals)
        assert all(s.instrument == "TEST" for s in signals)
        assert all(s.suggested_target > s.suggested_entry for s in signals)

    def test_no_signals_in_downtrend(self) -> None:
        bars = _bars([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        strat = SMACrossoverStrategy("TEST", fast=2, slow=3)
        assert list(strat.generate_signals(bars)) == []

    def test_no_lookahead_signal_uses_prior_close_state(self) -> None:
        # The signal for bar t reflects fast>slow as of t-1, never t.
        bars = _bars([1, 2, 3, 4, 5])
        strat = SMACrossoverStrategy("TEST", fast=2, slow=3, expected_move_lookback=2)
        closes = bars["Close"]
        long_prev = (closes.rolling(2).mean() > closes.rolling(3).mean()).shift(1)
        expected_ts = [
            ts.to_pydatetime() for ts in closes.index if bool(long_prev.loc[ts])
        ]
        signals = list(strat.generate_signals(bars))
        # every emitted signal day must be a day where prior-close crossover held
        assert {s.timestamp for s in signals}.issubset(set(expected_ts))

    def test_is_stateless_repeated_calls_identical(self) -> None:
        bars = _bars([1, 2, 3, 4, 5, 6, 7, 8])
        strat = SMACrossoverStrategy("TEST", fast=2, slow=3, expected_move_lookback=2)
        first = list(strat.generate_signals(bars))
        second = list(strat.generate_signals(bars))
        assert first == second

    def test_name_encodes_windows(self) -> None:
        assert SMACrossoverStrategy("TEST", fast=10, slow=30).name == "sma_crossover_10_30"
