"""Performance analysis for backtests: trade records, metrics, and reports.

This module defines the raw `Trade`/`EquityPoint` records the runner produces
and the `PerformanceReport` that scores a run. `build_performance_report` takes
primitives (not a `BacktestResult`) so this module never imports `runner` —
keeping the dependency one-directional and cycle-free.

Money/prices are `Decimal` (P&L, equity). Statistical ratios (Sharpe, drawdown,
win rate, ...) are `float` — they are pandas/numpy math, not domain money, which
CLAUDE.md explicitly allows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd
import structlog
from pydantic import BaseModel, ConfigDict

from packages.core.domain.signal import Direction

log = structlog.get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Trade:
    """A single executed round-trip trade.

    For the intraday open->close model, `entry_time == exit_time` (same day) and
    `reason` records why the position was closed.
    """

    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal  # underlying price at entry
    exit_price: Decimal  # underlying price at exit
    direction: Direction
    cert_return: Decimal  # net-of-cost return on the cert position (leveraged)
    pnl: Decimal  # money P&L in account currency
    reason: str


class EquityPoint(BaseModel):
    """One point on the equity curve."""

    model_config = ConfigDict(frozen=True)

    ts: datetime
    equity: Decimal


class PerformanceReport(BaseModel):
    """Scored result of a backtest. Boundary type (printed, serialized to PRs)."""

    model_config = ConfigDict(frozen=True)

    # Money
    initial_capital: Decimal
    final_equity: Decimal
    total_pnl: Decimal
    ev_per_trade: Decimal  # mean P&L per executed trade, in account currency

    # Ratios / statistics (fractions, not percentages)
    total_return: float
    buy_and_hold_return: float
    sharpe: float
    sortino: float
    max_drawdown: float  # positive magnitude, e.g. 0.15 == -15% peak-to-trough
    win_rate: float
    profit_factor: float  # gross profit / gross loss; inf if no losing trades

    # Counts
    trade_count: int
    signals_blocked: int

    equity_curve: list[EquityPoint]

    @property
    def excess_return_vs_baseline(self) -> float:
        """Total return minus buy-and-hold baseline (the edge, if positive)."""
        return self.total_return - self.buy_and_hold_return

    def summary(self) -> str:
        """Human-readable one-screen summary for CLI / PR descriptions."""

        def pct(x: float) -> str:
            return f"{x * 100:+.2f}%"

        pf = "inf" if self.profit_factor == float("inf") else f"{self.profit_factor:.2f}"
        lines = [
            "=== Backtest performance ===",
            f"Initial capital   : {self.initial_capital:,.2f}",
            f"Final equity      : {self.final_equity:,.2f}",
            f"Total P&L         : {self.total_pnl:,.2f}",
            f"Total return      : {pct(self.total_return)}",
            f"Buy & hold (base) : {pct(self.buy_and_hold_return)}",
            f"Excess vs base    : {pct(self.excess_return_vs_baseline)}",
            f"Sharpe (ann.)     : {self.sharpe:.2f}",
            f"Sortino (ann.)    : {self.sortino:.2f}",
            f"Max drawdown      : {pct(-self.max_drawdown)}",
            f"Win rate          : {pct(self.win_rate)}",
            f"Profit factor     : {pf}",
            f"Trades            : {self.trade_count}",
            f"EV / trade        : {self.ev_per_trade:,.2f}",
            f"Blocked by cost   : {self.signals_blocked}",
        ]
        return "\n".join(lines)


def _equity_floats(equity_curve: list[EquityPoint]) -> pd.Series:
    return pd.Series([float(p.equity) for p in equity_curve], dtype=float)


def _sharpe(returns: pd.Series, periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(returns.mean()) / std * float(periods_per_year**0.5)


def _sortino(returns: pd.Series, periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    dstd = float(downside.std(ddof=1))
    if dstd == 0.0:
        return 0.0
    return float(returns.mean()) / dstd * float(periods_per_year**0.5)


def _max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    worst = float(drawdown.min())
    return abs(worst)  # positive magnitude


def build_performance_report(
    *,
    trades: list[Trade],
    equity_curve: list[EquityPoint],
    signals_blocked: int,
    initial_capital: Decimal,
    buy_and_hold_return: Decimal,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PerformanceReport:
    """Score a backtest run into a `PerformanceReport`.

    Takes primitives rather than a `BacktestResult` so this module stays
    independent of `runner` (no import cycle).
    """
    final_equity = equity_curve[-1].equity if equity_curve else initial_capital
    total_pnl = final_equity - initial_capital
    total_return = (
        float(total_pnl / initial_capital) if initial_capital != 0 else 0.0
    )

    equity = _equity_floats(equity_curve)
    returns = equity.pct_change().dropna()
    sharpe = _sharpe(returns, periods_per_year)
    sortino = _sortino(returns, periods_per_year)
    max_dd = _max_drawdown(equity)

    trade_count = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    gross_profit = sum((t.pnl for t in wins), Decimal("0"))
    gross_loss = -sum((t.pnl for t in trades if t.pnl < 0), Decimal("0"))
    win_rate = (len(wins) / trade_count) if trade_count else 0.0
    if gross_loss > 0:
        profit_factor = float(gross_profit / gross_loss)
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    ev_per_trade = (total_pnl / trade_count) if trade_count else Decimal("0")

    report = PerformanceReport(
        initial_capital=initial_capital,
        final_equity=final_equity,
        total_pnl=total_pnl,
        ev_per_trade=ev_per_trade,
        total_return=total_return,
        buy_and_hold_return=float(buy_and_hold_return),
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        trade_count=trade_count,
        signals_blocked=signals_blocked,
        equity_curve=equity_curve,
    )
    log.info(
        "performance.report",
        trades=trade_count,
        blocked=signals_blocked,
        total_return=round(total_return, 4),
        sharpe=round(sharpe, 2),
        max_drawdown=round(max_dd, 4),
    )
    return report
