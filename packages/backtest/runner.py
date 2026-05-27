"""Backtest runner: drives a Strategy through the cost engine + cert simulator.

Hold model is **intraday open->close**: each bar the strategy is "long", we
enter at that bar's Open and exit at its Close as one same-day trade. This
honors the no-overnight hard rule (CLAUDE.md) and uses
`simulator.simulate_intraday_trade`. The strategy computes signals from prior
closes (no lookahead); the runner supplies the actual Open/Close fills.

Dependency rule (#6): this module imports only `core` and its own `backtest`
modules. It does NOT import `market_data` or `strategies`. Bars arrive through
an injected `BarsLoader`, and the strategy is any object satisfying the
structural `StrategyLike` protocol. The composition root (scripts/run_backtest.py)
wires the real `market_data.get_bars` + concrete strategy in. This is the
"only the data source + execution differ" seam from ARCHITECTURE.md.

Cost handling: the cost engine first **gates** each signal
(`signal_passes_cost_filter`). For executed trades we also **deduct** the
round-trip cost so reported EV/returns are net of costs (CLAUDE.md: "EV after
costs"). We deduct `estimate_round_trip_cost().total_pct` directly from the cert
return, i.e. as a fraction of the *cert* position. That matches CLAUDE.md's
"issuer spread 0.3-0.8% round-trip" (a cost on the cert you trade) and is
consistent with `required_underlying_move_for_breakeven` (= total_pct / leverage,
via `CostBreakdown.in_underlying_terms`), which the gating filter is built on.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Protocol

import pandas as pd
import structlog

from packages.backtest.cost_model import (
    estimate_round_trip_cost,
    signal_passes_cost_filter,
)
from packages.backtest.performance import EquityPoint, Trade
from packages.backtest.simulator import CertSpec, simulate_intraday_trade
from packages.core.domain.signal import Direction, Signal

log = structlog.get_logger(__name__)

DEFAULT_INTERVAL = "1d"


class BarsLoader(Protocol):
    """Loads OHLCV bars. Structurally matches `market_data.get_bars`."""

    def __call__(
        self,
        ticker: str,
        start: date | datetime,
        end: date | datetime,
        interval: str,
    ) -> pd.DataFrame: ...


class StrategyLike(Protocol):
    """Structural strategy contract — matches `strategies.base.Strategy`.

    Declared locally so `backtest` need not import the `strategies` package.
    """

    @property
    def name(self) -> str: ...

    def generate_signals(self, market_data: pd.DataFrame) -> Iterable[Signal]: ...


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a single backtest run."""

    instrument: str
    cert_leverage: Decimal  # magnitude, e.g. 5 for a 5x cert; direction applied per-trade
    start_date: date
    end_date: date
    initial_capital: Decimal
    position_size_pct: Decimal  # fraction of equity committed per trade, e.g. 0.2
    safety_margin: Decimal = Decimal("1.5")
    interval: str = DEFAULT_INTERVAL


@dataclass(frozen=True)
class BacktestResult:
    """Raw output of a run. Scored into a report by `performance`."""

    config: BacktestConfig
    trades: list[Trade]
    equity_curve: list[EquityPoint]
    signals_blocked: int
    buy_and_hold_return: Decimal  # unleveraged underlying return over the window
    bars_count: int = field(default=0)


class BacktestRunner:
    """Runs a Strategy over historical bars and records the result."""

    def __init__(
        self,
        config: BacktestConfig,
        strategy: StrategyLike,
        bars_loader: BarsLoader,
    ) -> None:
        self._config = config
        self._strategy = strategy
        self._load_bars = bars_loader

    def run(self) -> BacktestResult:
        cfg = self._config
        bars = self._load_bars(
            cfg.instrument, cfg.start_date, cfg.end_date, cfg.interval
        )
        if bars.empty:
            log.warning("backtest.no_bars", instrument=cfg.instrument)
            return BacktestResult(
                config=cfg,
                trades=[],
                equity_curve=[],
                signals_blocked=0,
                buy_and_hold_return=Decimal("0"),
                bars_count=0,
            )

        signals_by_ts: dict[pd.Timestamp, Signal] = {
            pd.Timestamp(sig.timestamp): sig
            for sig in self._strategy.generate_signals(bars)
        }

        equity = cfg.initial_capital
        trades: list[Trade] = []
        equity_curve: list[EquityPoint] = []
        blocked = 0
        # Round-trip cost as a fraction of the cert position (see module docstring).
        cost_pct = estimate_round_trip_cost().total_pct

        for ts, row in bars.iterrows():
            sig = signals_by_ts.get(pd.Timestamp(ts))
            if sig is not None:
                expected_move = self._expected_move(sig)
                if not signal_passes_cost_filter(
                    expected_underlying_move_pct=expected_move,
                    leverage=cfg.cert_leverage,
                    safety_margin=cfg.safety_margin,
                ):
                    blocked += 1
                else:
                    trade, equity = self._execute(sig, row, equity, cost_pct)
                    trades.append(trade)
            equity_curve.append(EquityPoint(ts=_as_datetime(ts), equity=equity))

        first_close = Decimal(str(bars["Close"].iloc[0]))
        last_close = Decimal(str(bars["Close"].iloc[-1]))
        baseline = (
            (last_close / first_close) - Decimal("1") if first_close else Decimal("0")
        )

        log.info(
            "backtest.done",
            instrument=cfg.instrument,
            bars=len(bars),
            trades=len(trades),
            blocked=blocked,
            final_equity=str(equity),
        )
        return BacktestResult(
            config=cfg,
            trades=trades,
            equity_curve=equity_curve,
            signals_blocked=blocked,
            buy_and_hold_return=baseline,
            bars_count=len(bars),
        )

    @staticmethod
    def _expected_move(sig: Signal) -> Decimal:
        """Expected favorable underlying move implied by the signal's levels."""
        return abs(sig.suggested_target - sig.suggested_entry) / sig.suggested_entry

    def _execute(
        self,
        sig: Signal,
        row: pd.Series,
        equity: Decimal,
        cost_pct: Decimal,
    ) -> tuple[Trade, Decimal]:
        cfg = self._config
        leverage = (
            cfg.cert_leverage
            if sig.direction == Direction.LONG
            else -cfg.cert_leverage
        )
        cert_spec = CertSpec(
            name=f"{cfg.instrument} {leverage}x",
            underlying=cfg.instrument,
            leverage=leverage,
        )
        entry = Decimal(str(row["Open"]))
        exit_price = Decimal(str(row["Close"]))
        _, gross_return = simulate_intraday_trade(entry, exit_price, cert_spec)
        net_return = gross_return - cost_pct  # deduct round-trip cost (cert terms)

        notional = equity * cfg.position_size_pct
        pnl = notional * net_return
        new_equity = equity + pnl

        ts = _as_datetime(sig.timestamp)
        trade = Trade(
            entry_time=ts,
            exit_time=ts,  # same-day: intraday open->close, no overnight hold
            entry_price=entry,
            exit_price=exit_price,
            direction=sig.direction,
            cert_return=net_return,
            pnl=pnl,
            reason="exit_on_close",
        )
        return trade, new_equity


def _as_datetime(ts: object) -> datetime:
    """Coerce a pandas Timestamp / datetime to a plain datetime."""
    if isinstance(ts, pd.Timestamp):
        dt = ts.to_pydatetime()
        assert isinstance(dt, datetime)  # narrow pandas Any -> datetime
        return dt
    assert isinstance(ts, datetime)
    return ts
