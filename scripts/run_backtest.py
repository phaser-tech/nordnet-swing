"""CLI: run the SMA crossover backtest and print a performance report.

Composition root for the backtest: this is where `market_data`, `strategies`,
and `backtest` are wired together (the runner itself imports none of its sibling
packages — see packages/backtest/runner.py).

Usage:
    uv run python scripts/run_backtest.py
    uv run python scripts/run_backtest.py --instrument ^OMX --leverage 5 --start 2018-01-01
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

# Make the repo root importable when run as a script (repo is not installed).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
from dotenv import load_dotenv

from packages.backtest.performance import build_performance_report
from packages.backtest.runner import BacktestConfig, BacktestRunner
from packages.market_data.historical import get_bars
from packages.strategies.sma_crossover import SMACrossoverStrategy

log = structlog.get_logger("run_backtest")


def configure_logging() -> None:
    """Console renderer in dev, JSON in production (LOG_FORMAT / LOG_LEVEL)."""
    log_format = os.environ.get("LOG_FORMAT", "console").strip().lower()
    level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").strip().upper(), logging.INFO)
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if log_format == "json"
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instrument", default="^OMX")
    p.add_argument("--leverage", type=Decimal, default=Decimal("5"))
    p.add_argument("--fast", type=int, default=10)
    p.add_argument("--slow", type=int, default=30)
    p.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date.today())
    p.add_argument("--capital", type=Decimal, default=Decimal("100000"))
    p.add_argument("--position-size", type=Decimal, default=Decimal("0.2"))
    p.add_argument("--safety-margin", type=Decimal, default=Decimal("1.5"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    configure_logging()
    args = parse_args(argv)

    config = BacktestConfig(
        instrument=args.instrument,
        cert_leverage=args.leverage,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        safety_margin=args.safety_margin,
    )
    strategy = SMACrossoverStrategy(args.instrument, fast=args.fast, slow=args.slow)
    runner = BacktestRunner(config, strategy, bars_loader=get_bars)

    log.info(
        "run_backtest.start",
        instrument=config.instrument,
        leverage=str(config.cert_leverage),
        start=str(config.start_date),
        end=str(config.end_date),
        strategy=strategy.name,
    )
    result = runner.run()
    report = build_performance_report(
        trades=result.trades,
        equity_curve=result.equity_curve,
        signals_blocked=result.signals_blocked,
        initial_capital=config.initial_capital,
        buy_and_hold_return=result.buy_and_hold_return,
    )
    print("\n" + report.summary() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
