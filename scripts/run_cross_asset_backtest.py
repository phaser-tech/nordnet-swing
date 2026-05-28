"""CLI: run the cross-asset confluence backtest and print a performance report.

Composition root: assembles the wide cross-asset frame (via market_data) and
injects it into the runner as the bars loader. The runner uses the OMX OHLCV
columns for fills and hands the whole wide frame to the strategy.

Usage:
    uv run python scripts/run_cross_asset_backtest.py
    uv run python scripts/run_cross_asset_backtest.py --k 0.6 --min-agree 4
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

import pandas as pd
import structlog
from dotenv import load_dotenv

from packages.backtest.assembly import assemble_cross_asset_frame
from packages.backtest.performance import build_performance_report
from packages.backtest.runner import BacktestConfig, BacktestRunner
from packages.market_data.historical import get_bars
from packages.strategies.cross_asset_confluence import (
    CrossAssetConfluenceStrategy,
)

log = structlog.get_logger("run_cross_asset_backtest")


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
    p.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date.today())
    p.add_argument("--capital", type=Decimal, default=Decimal("100000"))
    p.add_argument("--position-size", type=Decimal, default=Decimal("0.2"))
    p.add_argument("--safety-margin", type=Decimal, default=Decimal("1.5"))
    p.add_argument("--spx-sma", type=int, default=50)
    p.add_argument("--zwindow", type=int, default=60)
    p.add_argument("--k", type=float, default=0.5)
    p.add_argument("--min-agree", type=int, default=4)
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
    strategy = CrossAssetConfluenceStrategy(
        args.instrument,
        spx_sma_window=args.spx_sma,
        zscore_window=args.zwindow,
        deadband_k=args.k,
        min_agree=args.min_agree,
    )

    log.info(
        "cross_asset.start",
        instrument=config.instrument,
        leverage=str(config.cert_leverage),
        start=str(config.start_date),
        end=str(config.end_date),
        k=args.k,
        min_agree=args.min_agree,
    )

    wide = assemble_cross_asset_frame(
        loader=get_bars,
        instrument=config.instrument,
        start=config.start_date,
        end=config.end_date,
        interval=config.interval,
    )

    def wide_loader(ticker: str, start: object, end: object, interval: str) -> pd.DataFrame:
        return wide

    runner = BacktestRunner(config, strategy, bars_loader=wide_loader)
    result = runner.run()
    report = build_performance_report(
        trades=result.trades,
        equity_curve=result.equity_curve,
        signals_blocked=result.signals_blocked,
        initial_capital=config.initial_capital,
        buy_and_hold_return=result.buy_and_hold_return,
    )

    print("\n" + report.summary())
    # Cadence: how selective is the gate? (~5 trading days per week)
    weeks = result.bars_count / 5 if result.bars_count else 0
    n_trades = len(result.trades)
    if weeks:
        print(f"Trade frequency   : {n_trades / weeks:.2f}/week ({n_trades} over ~{weeks:.0f} weeks)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
