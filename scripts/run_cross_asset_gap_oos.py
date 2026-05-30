"""CLI: OOS validation for the cross-asset gap-capture strategy (#21).

Apples-to-apples vs PR #10: SAME signal source (frozen `CrossAssetConfluenceStrategy`
defaults), SAME instrument (`^OMX`), SAME split (2023-01-01), SAME 5x leverage.
The ONLY change is the trade horizon: close(T-1) -> open(T) instead of
open(T) -> close(T). So any difference is attributable to capturing the
overnight gap the four open->close strategies couldn't reach.

Cost includes the one-night overnight financing term added in commit 1 on this
branch (`estimate_round_trip_cost(overnight_nights=1)`), per the ratified
exception (#21, CLAUDE.md "Approved overnight exceptions").

Usage:
    uv run python scripts/run_cross_asset_gap_oos.py
    uv run python scripts/run_cross_asset_gap_oos.py --split 2023-01-01
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import structlog
from dotenv import load_dotenv

from packages.backtest.assembly import assemble_cross_asset_frame
from packages.backtest.cost_model import estimate_round_trip_cost
from packages.backtest.oos import SetStats, run_oos
from packages.core.domain.signal import Direction
from packages.market_data.historical import get_bars
from packages.strategies.cross_asset_confluence import CrossAssetConfluenceStrategy
from packages.strategies.cross_asset_gap import (
    GAP_DIRECTION_COL,
    CrossAssetGapStrategy,
    build_per_day_gap,
)

log = structlog.get_logger("run_cross_asset_gap_oos")

LEVERAGE = 5.0  # FIXED. Same as PR #10 -- this is an apples-to-apples test.
INSTRUMENT = "^OMX"


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
    """Pre-registered decision tree from issue #21."""
    gross_holds = test.t_stat >= 2.0 and test.mean_signed_move > 0
    net_holds = test.net_return > 0 and test.net_sharpe > 0
    if gross_holds and net_holds:
        return (
            "Branch A — NET EDGE HOLDS OOS. Gap-arbitrage hypothesis empirically "
            "vindicated AND we have a tradeable strategy. Validate at conservative "
            "financing assumptions; queue Phase 1.5."
        )
    if test.mean_signed_move > 0:
        return (
            "Branch B — fails OOS, but the gap mean is positive in the signal "
            "direction (just not significant). The gap is real but too small to "
            "net-clear at 5x and one-night financing cost. Decide whether to keep "
            "the carve-out for higher-leverage / lower-cost scenarios."
        )
    return (
        "Branch C — fails OOS AND the gap mean is non-positive in the signal "
        "direction. The original gap-arbitrage diagnostic was wrong: the moves we "
        "observed in the gap weren't aligned with the cross-asset signal. The "
        "right next step is a structurally different SIGNAL SOURCE (event-window, "
        "regime gate, calendar effect), not another horizon experiment."
    )


def _print_signal_breakdown(name: str, frame: pd.DataFrame) -> None:
    counts: Counter[str] = Counter()
    for v in frame[GAP_DIRECTION_COL]:
        if v is Direction.LONG:
            counts["LONG"] += 1
        elif v is Direction.SHORT:
            counts["SHORT"] += 1
        else:
            counts["no-trade"] += 1
    total = sum(counts.values())
    trade = counts["LONG"] + counts["SHORT"]
    rate = trade * 100 / total if total else 0.0
    print(
        f"{name:<6}{total:>6} days  LONG={counts['LONG']:>3}  SHORT={counts['SHORT']:>3}  "
        f"no-trade={counts['no-trade']:>3}  (trade rate {rate:.0f}%)"
    )


def _print_gap_distribution(name: str, frame: pd.DataFrame) -> None:
    """Per-direction signed gap return distribution -- the raw observation
    behind the OOS mean."""
    for direction, label in ((Direction.LONG, "LONG"), (Direction.SHORT, "SHORT")):
        sign = 1.0 if direction is Direction.LONG else -1.0
        signed_moves: list[float] = []
        for _, row in frame.iterrows():
            if row[GAP_DIRECTION_COL] is direction:
                open_ = float(row["Open"])
                close_ = float(row["Close"])
                signed_moves.append(sign * (close_ - open_) / open_)
        n = len(signed_moves)
        if n == 0:
            print(f"  {name:<6} {label:<6}  (no signals)")
            continue
        mean = statistics.fmean(signed_moves)
        std = statistics.stdev(signed_moves) if n > 1 else 0.0
        t = mean / (std / (n**0.5)) if std > 0 else 0.0
        hit = sum(1 for m in signed_moves if m > 0) / n
        print(
            f"  {name:<6} {label:<6}  n={n:>3}  mean={mean * 100:>+7.3f}%  "
            f"std={std * 100:>5.3f}%  t={t:>+5.2f}  hit={hit * 100:>4.1f}%"
        )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    configure_logging()
    args = parse_args(argv)

    # Cost includes one-night overnight financing per the ratified exception.
    cost_breakdown = estimate_round_trip_cost(overnight_nights=1)
    cost_pct = float(cost_breakdown.total_pct)
    breakeven_underlying = cost_pct / LEVERAGE

    wide = assemble_cross_asset_frame(
        loader=get_bars,
        instrument=INSTRUMENT,
        start=args.start,
        end=args.end,
        interval="1d",
    )
    if wide.empty:
        log.error("no_bars", instrument=INSTRUMENT)
        return 1

    # Same frozen config as PR #10 -- defaults.
    confluence = CrossAssetConfluenceStrategy(INSTRUMENT)
    signals = list(confluence.generate_signals(wide))
    synthetic = build_per_day_gap(wide, signals)
    if synthetic.empty:
        log.error("no_per_day_rows", instrument=INSTRUMENT)
        return 1

    strategy = CrossAssetGapStrategy(INSTRUMENT)
    train, test = run_oos(
        wide_frame=synthetic,
        strategy=strategy,
        split=args.split,
        leverage=LEVERAGE,
        cost_pct=cost_pct,
    )

    print("\n=== OOS validation (cross-asset gap-capture, ^OMX, 5x, one night) ===")
    print(
        f"Window: {wide.index.min().date()} -> {wide.index.max().date()}  "
        f"Split: {args.split}"
    )
    print(
        f"Round-trip cost {cost_pct * 100:.2f}% cert "
        f"(spread {float(cost_breakdown.spread_pct) * 100:.2f}% + "
        f"slippage {float(cost_breakdown.slippage_pct) * 100:.2f}% + "
        f"1-night financing {float(cost_breakdown.overnight_financing_pct) * 100:.2f}%)  "
        f"/ breakeven {breakeven_underlying * 100:.3f}% underlying (per gap)\n"
    )
    header = (
        f"{'set':<6}{'trades':>8}{'win%':>9}{'meanMove%':>13}{'t-stat':>8}"
        f"{'gross%':>11}{'net%':>11}{'grSh':>9}{'netSh':>9}"
    )
    print(header)
    print(_row(train))
    print(_row(test))
    print("\nVERDICT (pre-registered tree from #21): " + _verdict(test))

    train_frame = synthetic[synthetic.index < pd.Timestamp(args.split, tz=synthetic.index.tz)]
    test_frame = synthetic[synthetic.index >= pd.Timestamp(args.split, tz=synthetic.index.tz)]

    print("\n=== Signal breakdown ===")
    _print_signal_breakdown("train", train_frame)
    _print_signal_breakdown("test ", test_frame)
    _print_signal_breakdown("all  ", synthetic)

    print("\n=== Gap signed-move distribution per direction ===")
    _print_gap_distribution("train", train_frame)
    _print_gap_distribution("test ", test_frame)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
