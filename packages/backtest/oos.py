"""Out-of-sample (walk-forward) validation harness.

Answers one question: does the strategy's net intraday edge hold out-of-sample?

Design (matches issue #9):
- Config is FROZEN. Signals are computed once on the continuous wide frame
  (trailing z-windows span the train/test boundary, as live trading would),
  then partitioned by date. The test set is never used to select parameters.
- Both GROSS (no cost) and NET (after round-trip cost) are reported per set.
- Fixed leverage; no sweeping here.

Returns are computed in float (analytics layer, not the trading domain). The
per-trade model matches the runner: cert_return = leverage * signed_underlying
intraday move; net subtracts the cert-terms round-trip cost.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date

import pandas as pd

from packages.backtest.cost_model import estimate_round_trip_cost
from packages.backtest.runner import StrategyLike
from packages.core.domain.signal import Direction, Signal

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class SetStats:
    """Per-period (train or test) evaluation stats."""

    name: str
    n_trades: int
    win_rate: float  # directional hit rate: signed intraday move > 0
    mean_signed_move: float  # mean signed underlying intraday move (fraction)
    t_stat: float  # significance of the directional edge
    gross_return: float  # total return, no cost
    net_return: float  # total return, after round-trip cost
    gross_sharpe: float  # annualized, daily, no cost
    net_sharpe: float  # annualized, daily, after cost


def _daily_returns(equity_curve: list[float]) -> list[float]:
    return [
        equity_curve[i] / equity_curve[i - 1] - 1.0
        for i in range(1, len(equity_curve))
        if equity_curve[i - 1] != 0
    ]


def _sharpe(daily_returns: list[float], periods_per_year: int) -> float:
    if len(daily_returns) < 2:
        return 0.0
    std = statistics.stdev(daily_returns)
    if std == 0.0:
        return 0.0
    return statistics.fmean(daily_returns) / std * math.sqrt(periods_per_year)


def _evaluate(
    name: str,
    bars: pd.DataFrame,
    signals_by_ts: dict[pd.Timestamp, Signal],
    *,
    leverage: float,
    cost_pct: float,
    position_size_pct: float,
    periods_per_year: int,
) -> SetStats:
    """Evaluate one period: walk its bars, apply in-range signals intraday."""
    signed_moves: list[float] = []
    gross_equity = 1.0
    net_equity = 1.0
    gross_curve: list[float] = []
    net_curve: list[float] = []

    for ts, row in bars.iterrows():
        sig = signals_by_ts.get(pd.Timestamp(ts))
        if sig is not None:
            open_ = float(row["Open"])
            close = float(row["Close"])
            sign = 1.0 if sig.direction == Direction.LONG else -1.0
            move = sign * (close - open_) / open_  # signed intraday underlying move
            signed_moves.append(move)
            gross_cert = leverage * move
            net_cert = gross_cert - cost_pct
            gross_equity *= 1.0 + position_size_pct * gross_cert
            net_equity *= 1.0 + position_size_pct * net_cert
        gross_curve.append(gross_equity)
        net_curve.append(net_equity)

    n = len(signed_moves)
    win_rate = sum(1 for m in signed_moves if m > 0) / n if n else 0.0
    mean_move = statistics.fmean(signed_moves) if n else 0.0
    if n > 1 and statistics.stdev(signed_moves) > 0:
        t_stat = mean_move / (statistics.stdev(signed_moves) / math.sqrt(n))
    else:
        t_stat = 0.0

    return SetStats(
        name=name,
        n_trades=n,
        win_rate=win_rate,
        mean_signed_move=mean_move,
        t_stat=t_stat,
        gross_return=gross_equity - 1.0,
        net_return=net_equity - 1.0,
        gross_sharpe=_sharpe(_daily_returns(gross_curve), periods_per_year),
        net_sharpe=_sharpe(_daily_returns(net_curve), periods_per_year),
    )


def run_oos(
    *,
    wide_frame: pd.DataFrame,
    strategy: StrategyLike,
    split: date,
    leverage: float,
    position_size_pct: float = 0.2,
    cost_pct: float | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> tuple[SetStats, SetStats]:
    """Run the frozen strategy over train (< split) and test (>= split).

    Signals are computed ONCE on the full frame so trailing windows span the
    boundary; only the date partition differs between sets. Returns
    (train_stats, test_stats).
    """
    if cost_pct is None:
        cost_pct = float(estimate_round_trip_cost().total_pct)

    signals_by_ts = {
        pd.Timestamp(s.timestamp): s for s in strategy.generate_signals(wide_frame)
    }
    split_ts = pd.Timestamp(split, tz=wide_frame.index.tz)

    train_bars = wide_frame[wide_frame.index < split_ts]
    test_bars = wide_frame[wide_frame.index >= split_ts]
    train_sigs = {ts: s for ts, s in signals_by_ts.items() if ts < split_ts}
    test_sigs = {ts: s for ts, s in signals_by_ts.items() if ts >= split_ts}

    train = _evaluate(
        "train",
        train_bars,
        train_sigs,
        leverage=leverage,
        cost_pct=cost_pct,
        position_size_pct=position_size_pct,
        periods_per_year=periods_per_year,
    )
    test = _evaluate(
        "test",
        test_bars,
        test_sigs,
        leverage=leverage,
        cost_pct=cost_pct,
        position_size_pct=position_size_pct,
        periods_per_year=periods_per_year,
    )
    return train, test
