"""Unit tests for the out-of-sample validation harness. Fixtures only."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from packages.backtest.oos import run_oos
from packages.core.domain.signal import Conviction, Direction, Signal

# Three train days (2022) + three test days (2023), split at 2023-01-01.
_DATES = [
    "2022-12-28",
    "2022-12-29",
    "2022-12-30",
    "2023-01-02",
    "2023-01-03",
    "2023-01-04",
]
_OPEN = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
_CLOSE = [102.0, 99.0, 100.0, 101.0, 100.5, 100.0]  # intraday: +2%, -1%, 0, +1%, +0.5%, 0


def _frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(_DATES), name="ts").tz_localize("UTC")
    return pd.DataFrame(
        {
            "Open": _OPEN,
            "High": [c + 1 for c in _CLOSE],
            "Low": [o - 1 for o in _OPEN],
            "Close": _CLOSE,
            "Volume": [1000] * len(_DATES),
        },
        index=idx,
    )


def _long(ts: pd.Timestamp) -> Signal:
    return Signal(
        timestamp=ts.to_pydatetime(),
        strategy_name="fake",
        instrument="TEST",
        direction=Direction.LONG,
        conviction=Conviction.HIGH,
        suggested_entry=Decimal("100"),
        suggested_stop=Decimal("99"),
        suggested_target=Decimal("102"),
    )


class _FakeStrategy:
    """Yields LONG signals on given index positions (satisfies StrategyLike)."""

    def __init__(self, positions: list[int]) -> None:
        self._positions = positions

    @property
    def name(self) -> str:
        return "fake"

    def generate_signals(self, market_data: pd.DataFrame) -> Iterable[Signal]:
        return [_long(market_data.index[p]) for p in self._positions]


# Signals on 2022-12-28, 2022-12-29 (train) and 2023-01-02, 2023-01-03 (test).
_STRATEGY_POSITIONS = [0, 1, 3, 4]
_SPLIT = date(2023, 1, 1)
_KW = dict(leverage=5.0, position_size_pct=0.2, cost_pct=0.006)


def _run() -> tuple:  # type: ignore[type-arg]
    return run_oos(
        wide_frame=_frame(),
        strategy=_FakeStrategy(_STRATEGY_POSITIONS),
        split=_SPLIT,
        **_KW,  # type: ignore[arg-type]
    )


class TestSplit:
    def test_partitions_signals_by_date(self) -> None:
        train, test = _run()
        assert train.n_trades == 2  # 2022 signals
        assert test.n_trades == 2  # 2023 signals


class TestTrainMetrics:
    def test_directional_stats(self) -> None:
        train, _ = _run()
        # moves +2%, -1% -> win 1/2, mean +0.5%
        assert train.win_rate == pytest.approx(0.5)
        assert train.mean_signed_move == pytest.approx(0.005)
        assert train.t_stat == pytest.approx(0.3333, abs=1e-3)

    def test_gross_and_net_returns(self) -> None:
        train, _ = _run()
        # gross: (1+0.2*0.10)*(1+0.2*-0.05) - 1
        assert train.gross_return == pytest.approx(1.02 * 0.99 - 1.0)
        # net: cost 0.006 per trade off the cert return
        assert train.net_return == pytest.approx(1.0188 * 0.9888 - 1.0, abs=1e-9)


class TestTestMetrics:
    def test_gross_and_net_returns(self) -> None:
        _, test = _run()
        # moves +1%, +0.5% -> gross cert 0.05, 0.025
        assert test.win_rate == pytest.approx(1.0)
        assert test.gross_return == pytest.approx(1.01 * 1.005 - 1.0)
        assert test.net_return == pytest.approx(1.0088 * 1.0038 - 1.0, abs=1e-9)


class TestCostDrag:
    def test_net_below_gross_when_trades_exist(self) -> None:
        train, test = _run()
        assert train.net_return < train.gross_return
        assert test.net_return < test.gross_return


class TestEmptySet:
    def test_no_signals_in_test_is_well_defined(self) -> None:
        # Split after all data -> test set has no signals.
        train, test = run_oos(
            wide_frame=_frame(),
            strategy=_FakeStrategy(_STRATEGY_POSITIONS),
            split=date(2030, 1, 1),
            **_KW,  # type: ignore[arg-type]
        )
        assert test.n_trades == 0
        assert test.gross_return == 0.0
        assert test.net_return == 0.0
        assert test.t_stat == 0.0
        assert train.n_trades == 4  # all signals now in train
