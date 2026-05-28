"""CLI: out-of-sample validation of the volume-confirmed breakout strategy.

Same frozen/blind discipline and fixed 5x as the cross-asset OOS run. Uses only
^OMX OHLCV (no cross-asset frame). Reuses packages/backtest/oos.py.

Usage:
    uv run python scripts/run_breakout_oos.py
    uv run python scripts/run_breakout_oos.py --split 2023-01-01
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

from packages.backtest.cost_model import estimate_round_trip_cost
from packages.backtest.oos import SetStats, run_oos
from packages.market_data.historical import get_bars
from packages.strategies.breakout_confluence import BreakoutConfluenceStrategy

log = structlog.get_logger("run_breakout_oos")

LEVERAGE = 5.0  # FIXED. No leverage sweep (Phase 1 question).


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


def _verdict(test: SetStats) -> str:
    gross_holds = test.t_stat >= 2.0 and test.mean_signed_move > 0
    net_holds = test.net_return > 0 and test.net_sharpe > 0
    if net_holds and gross_holds:
        return "NET EDGE HOLDS OOS — candidate for Phase 1 intraday validation."
    if gross_holds and not net_holds:
        return "GROSS survives OOS but NET does not — costs eat it. Not tradeable at 5x."
    return "NO EDGE OOS — daily-bar edge exhausted in Phase 0; next step is Phase 1."


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    configure_logging()
    args = parse_args(argv)

    cost_pct = float(estimate_round_trip_cost().total_pct)
    bars = get_bars(args.instrument, args.start, args.end, "1d")
    strategy = BreakoutConfluenceStrategy(args.instrument)  # frozen defaults

    log.info(
        "breakout_oos.start",
        instrument=args.instrument,
        train=f"{args.start}..{args.split}",
        test=f"{args.split}..{args.end}",
        leverage=LEVERAGE,
    )
    train, test = run_oos(
        wide_frame=bars, strategy=strategy, split=args.split, leverage=LEVERAGE
    )

    print("\n=== OOS validation (20d volume-confirmed breakout, 5x, frozen) ===")
    print(f"Train: {args.start} -> {args.split}    Test: {args.split} -> {args.end}")
    print(f"Round-trip cost {cost_pct * 100:.2f}% cert / breakeven {cost_pct / LEVERAGE * 100:.3f}% underlying\n")
    print(
        f"{'set':<6}{'trades':>8}{'win%':>9}{'meanMove%':>13}{'t-stat':>8}"
        f"{'gross%':>11}{'net%':>11}{'grSh':>9}{'netSh':>9}"
    )
    print(_row(train))
    print(_row(test))
    print("\nVERDICT: " + _verdict(test) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
