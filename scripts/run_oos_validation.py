"""CLI: out-of-sample validation of the cross-asset confluence strategy.

Composition root: assembles the wide frame, runs the FROZEN strategy over
train (< split) and test (>= split), prints a per-set gross/net table and an
honest verdict on whether the net intraday edge holds out-of-sample.

Fixed 5x leverage. No tuning on the test set.

Usage:
    uv run python scripts/run_oos_validation.py
    uv run python scripts/run_oos_validation.py --split 2023-01-01
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
from dotenv import load_dotenv

from packages.backtest.assembly import assemble_cross_asset_frame
from packages.backtest.cost_model import estimate_round_trip_cost
from packages.backtest.oos import SetStats, run_oos
from packages.market_data.historical import get_bars
from packages.strategies.cross_asset_confluence import (
    CrossAssetConfluenceStrategy,
)

log = structlog.get_logger("run_oos_validation")

LEVERAGE = 5.0  # FIXED. Leverage sweep is a Phase 1 question (needs real spreads).


def configure_logging() -> None:
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
    p.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date.today())
    p.add_argument("--split", type=date.fromisoformat, default=date(2023, 1, 1))
    return p.parse_args(argv)


def _row(s: SetStats) -> str:
    return (
        f"{s.name:<6}{s.n_trades:>8}{s.win_rate * 100:>9.1f}"
        f"{s.mean_signed_move * 100:>13.3f}{s.t_stat:>8.2f}"
        f"{s.gross_return * 100:>11.1f}{s.net_return * 100:>11.1f}"
        f"{s.gross_sharpe:>9.2f}{s.net_sharpe:>9.2f}"
    )


def _verdict(test: SetStats, breakeven_underlying: float) -> str:
    gross_holds = test.t_stat >= 2.0 and test.mean_signed_move > 0
    net_holds = test.net_return > 0 and test.net_sharpe > 0
    if net_holds and gross_holds:
        return "NET EDGE HOLDS OOS — gross edge significant AND net is positive."
    if gross_holds and not net_holds:
        return (
            "GROSS edge survives OOS but NET does not — costs eat it. "
            "Not tradeable at 5x as-is."
        )
    return "NO EDGE OOS — the gross directional edge does not survive. Stop here."


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    configure_logging()
    args = parse_args(argv)

    cost_pct = float(estimate_round_trip_cost().total_pct)
    breakeven_underlying = cost_pct / LEVERAGE

    wide = assemble_cross_asset_frame(
        loader=get_bars,
        instrument=args.instrument,
        start=args.start,
        end=args.end,
        interval="1d",
    )
    # Frozen config: defaults only. No tuning here, none on the test set.
    strategy = CrossAssetConfluenceStrategy(args.instrument)

    log.info(
        "oos.start",
        instrument=args.instrument,
        train=f"{args.start}..{args.split}",
        test=f"{args.split}..{args.end}",
        leverage=LEVERAGE,
    )
    train, test = run_oos(
        wide_frame=wide, strategy=strategy, split=args.split, leverage=LEVERAGE
    )

    print("\n=== Out-of-sample validation (cross-asset confluence, 5x, frozen) ===")
    print(f"Train: {args.start} -> {args.split}    Test: {args.split} -> {args.end}")
    print(f"Round-trip cost {cost_pct * 100:.2f}% cert / breakeven {breakeven_underlying * 100:.3f}% underlying\n")
    header = (
        f"{'set':<6}{'trades':>8}{'win%':>9}{'meanMove%':>13}{'t-stat':>8}"
        f"{'gross%':>11}{'net%':>11}{'grSh':>9}{'netSh':>9}"
    )
    print(header)
    print(_row(train))
    print(_row(test))
    print("\nVERDICT: " + _verdict(test, breakeven_underlying) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
