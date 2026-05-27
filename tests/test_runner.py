"""Unit tests for the backtest runner. Uses an injected loader; no DB/network."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from decimal import Decimal

import pandas as pd

from packages.backtest.runner import BacktestConfig, BacktestRunner
from packages.core.domain.signal import Conviction, Direction, Signal


def _bars() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), name="ts"
    ).tz_localize("UTC")
    return pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [103.0, 102.0, 104.0],
            "Low": [99.0, 99.0, 99.0],
            "Close": [102.0, 101.0, 103.0],
            "Volume": [1000, 1000, 1000],
        },
        index=idx,
    )


class _FixedStrategy:
    """Yields a predetermined list of signals (satisfies StrategyLike)."""

    def __init__(self, signals: list[Signal]) -> None:
        self._signals = signals

    @property
    def name(self) -> str:
        return "fixed"

    def generate_signals(self, market_data: pd.DataFrame) -> Iterable[Signal]:
        return iter(self._signals)


def _signal(ts: pd.Timestamp, entry: str, target: str) -> Signal:
    e = Decimal(entry)
    return Signal(
        timestamp=ts.to_pydatetime(),
        strategy_name="fixed",
        instrument="TEST",
        direction=Direction.LONG,
        conviction=Conviction.MEDIUM,
        suggested_entry=e,
        suggested_stop=e * Decimal("0.99"),
        suggested_target=Decimal(target),
    )


def _config() -> BacktestConfig:
    return BacktestConfig(
        instrument="TEST",
        cert_leverage=Decimal("5"),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 2, 1),
        initial_capital=Decimal("100000"),
        position_size_pct=Decimal("0.2"),
        safety_margin=Decimal("1.5"),
    )


def _loader(bars: pd.DataFrame):  # type: ignore[no-untyped-def]
    def load(ticker: str, start: object, end: object, interval: str) -> pd.DataFrame:
        return bars

    return load


def test_executes_passing_signal_and_blocks_cheap_one() -> None:
    bars = _bars()
    idx = bars.index
    signals = [
        _signal(idx[0], "100", "102"),  # expected move 2% -> passes cost filter
        _signal(idx[1], "100", "100.05"),  # 0.05% -> blocked (breakeven*1.5 = 0.18%)
    ]
    runner = BacktestRunner(_config(), _FixedStrategy(signals), bars_loader=_loader(bars))
    result = runner.run()

    assert len(result.trades) == 1
    assert result.signals_blocked == 1
    assert len(result.equity_curve) == 3  # one point per bar


def test_trade_pnl_and_equity_math() -> None:
    bars = _bars()
    idx = bars.index
    runner = BacktestRunner(
        _config(),
        _FixedStrategy([_signal(idx[0], "100", "102")]),
        bars_loader=_loader(bars),
    )
    result = runner.run()

    # Open=100, Close=102 -> underlying +2%, 5x = +10% gross, minus 0.6% cost = 9.4%.
    # notional = 100000 * 0.2 = 20000 -> pnl = 20000 * 0.094 = 1880.
    trade = result.trades[0]
    assert abs(trade.pnl - Decimal("1880")) < Decimal("0.01")
    assert abs(trade.cert_return - Decimal("0.094")) < Decimal("1e-9")
    final_equity = result.equity_curve[-1].equity
    assert abs(final_equity - Decimal("101880")) < Decimal("0.01")


def test_no_overnight_every_trade_is_same_day() -> None:
    bars = _bars()
    idx = bars.index
    runner = BacktestRunner(
        _config(),
        _FixedStrategy([_signal(idx[0], "100", "102"), _signal(idx[2], "100", "104")]),
        bars_loader=_loader(bars),
    )
    result = runner.run()
    assert result.trades
    assert all(t.entry_time == t.exit_time for t in result.trades)


def test_buy_and_hold_baseline_from_first_and_last_close() -> None:
    bars = _bars()
    runner = BacktestRunner(_config(), _FixedStrategy([]), bars_loader=_loader(bars))
    result = runner.run()
    # first Close 102, last Close 103
    assert abs(result.buy_and_hold_return - (Decimal("103") / Decimal("102") - 1)) < Decimal("1e-9")


def test_empty_bars_returns_empty_result() -> None:
    empty = _bars().iloc[0:0]
    runner = BacktestRunner(_config(), _FixedStrategy([]), bars_loader=_loader(empty))
    result = runner.run()
    assert result.trades == []
    assert result.equity_curve == []
    assert result.buy_and_hold_return == Decimal("0")
