"""CLI: sync historical daily bars for the default instruments into TimescaleDB.

Runs schema migrations (idempotent), then fetches + stores ~10 years of daily
bars for the default cross-asset universe so the backtest framework has data
to work with.

Usage:
    uv run python scripts/sync_market_data.py
    uv run python scripts/sync_market_data.py --years 5
    uv run python scripts/sync_market_data.py --tickers ^OMXS30 ^NDX
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Make the repo root importable when run as a script (python scripts/...py),
# since this repo is not installed as a package ([tool.uv] package = false).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
from dotenv import load_dotenv

from packages.market_data.db import run_migrations
from packages.market_data.historical import (
    DAILY_INTERVAL,
    DEFAULT_INSTRUMENTS,
    sync,
)

log = structlog.get_logger("sync_market_data")


def configure_logging() -> None:
    """Configure structlog: console renderer in dev, JSON in production.

    Controlled by ``LOG_FORMAT`` (``console``|``json``) and ``LOG_LEVEL``.
    Entry points configure logging; library modules only ``get_logger``.
    """
    log_format = os.environ.get("LOG_FORMAT", "console").strip().lower()
    level_name = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="How many years of history to fetch (default: 10).",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_INSTRUMENTS),
        help="Tickers to sync (default: the cross-asset universe).",
    )
    parser.add_argument(
        "--interval",
        default=DAILY_INTERVAL,
        help=f"Bar interval (default: {DAILY_INTERVAL}; only daily supported).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    configure_logging()
    args = parse_args(argv)

    start_date = date.today() - timedelta(days=365 * args.years)
    log.info(
        "sync.start",
        tickers=args.tickers,
        years=args.years,
        start=str(start_date),
        interval=args.interval,
    )

    run_migrations()
    results = sync(tuple(args.tickers), start_date, interval=args.interval)

    for ticker, count in results.items():
        log.info("sync.result", ticker=ticker, rows=count)
    log.info("sync.complete", total_rows=sum(results.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
