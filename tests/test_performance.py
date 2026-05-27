"""Unit tests for the performance analyzer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from packages.backtest.performance import (
    EquityPoint,
    Trade,
    build_performance_report,
)
from packages.core.domain.signal import Direction


def _trade(pnl: str) -> Trade:
    ts = datetime(2024, 1, 2, tzinfo=UTC)
    return Trade(
        entry_time=ts,
        exit_time=ts,
        entry_price=Decimal("100"),
        exit_price=Decimal("101"),
        direction=Direction.LONG,
        cert_return=Decimal("0.05"),
        pnl=Decimal(pnl),
        reason="exit_on_close",
    )


def _curve(values: list[str]) -> list[EquityPoint]:
    base = datetime(2024, 1, 2, tzinfo=UTC)
    return [
        EquityPoint(ts=base + timedelta(days=i), equity=Decimal(v))
        for i, v in enumerate(values)
    ]


class TestBasicMetrics:
    def _report(self):  # type: ignore[no-untyped-def]
        return build_performance_report(
            trades=[_trade("1000"), _trade("-500"), _trade("1500")],
            equity_curve=_curve(["100000", "101000", "100500", "102000"]),
            signals_blocked=2,
            initial_capital=Decimal("100000"),
            buy_and_hold_return=Decimal("0.01"),
        )

    def test_money_aggregates(self) -> None:
        r = self._report()
        assert r.final_equity == Decimal("102000")
        assert r.total_pnl == Decimal("2000")
        assert r.total_return == pytest.approx(0.02)

    def test_trade_stats(self) -> None:
        r = self._report()
        assert r.trade_count == 3
        assert r.win_rate == pytest.approx(2 / 3)
        assert r.profit_factor == pytest.approx(5.0)  # 2500 / 500
        assert r.ev_per_trade == pytest.approx(Decimal("2000") / 3)

    def test_risk_stats(self) -> None:
        r = self._report()
        assert r.max_drawdown == pytest.approx(0.0049505, abs=1e-4)  # 100500/101000-1
        assert r.sharpe > 0  # net positive drift
        assert r.sortino >= 0

    def test_passthrough_and_baseline(self) -> None:
        r = self._report()
        assert r.signals_blocked == 2
        assert r.buy_and_hold_return == pytest.approx(0.01)
        assert r.excess_return_vs_baseline == pytest.approx(0.01)

    def test_summary_renders(self) -> None:
        s = self._report().summary()
        assert "Total return" in s
        assert "Blocked by cost" in s


class TestEdgeCases:
    def test_profit_factor_inf_when_no_losses(self) -> None:
        r = build_performance_report(
            trades=[_trade("1000"), _trade("500")],
            equity_curve=_curve(["100000", "101000", "101500"]),
            signals_blocked=0,
            initial_capital=Decimal("100000"),
            buy_and_hold_return=Decimal("0"),
        )
        assert r.profit_factor == float("inf")

    def test_no_trades_is_well_defined(self) -> None:
        r = build_performance_report(
            trades=[],
            equity_curve=_curve(["100000", "100000", "100000"]),
            signals_blocked=5,
            initial_capital=Decimal("100000"),
            buy_and_hold_return=Decimal("0.03"),
        )
        assert r.trade_count == 0
        assert r.win_rate == 0.0
        assert r.profit_factor == 0.0
        assert r.ev_per_trade == Decimal("0")
        assert r.total_return == pytest.approx(0.0)
        assert r.sharpe == 0.0  # zero-variance equity
