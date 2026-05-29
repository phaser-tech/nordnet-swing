"""CLI: OOS validation + diagnostics for the opening-range break strategy (#19).

Composition root. Loads the `^OMX` 1h bars ingested in #17, builds the synthetic
per-day ORB frame, runs the FROZEN strategy through `oos.py`, then prints the
diagnostic decompositions called out in the issue:

  - Break-frequency split  : LONG / SHORT / no-trade per period
  - Hold-length distribution
  - Time-of-day of break confirmations
  - First-bar (range bar) return distribution -- tells us whether the "range"
    is essentially the overnight gap re-priced

The OOS verdict scores `sign * (close - open) / open` on the synthetic frame --
i.e. the actual entry-bar-open → last-full-bar-close move for break days.
Frozen 5x leverage, default cost. No tuning on the test set.

Usage:
    uv run python scripts/run_orb_oos.py
    uv run python scripts/run_orb_oos.py --split 2025-12-01 --min-trailing 3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import structlog
from dotenv import load_dotenv

from packages.backtest.cost_model import estimate_round_trip_cost
from packages.backtest.oos import SetStats, run_oos
from packages.core.domain.signal import Direction
from packages.market_data.historical import HOURLY_INTERVAL, get_bars
from packages.strategies.opening_range_break import (
    ORB_BARS_HELD_COL,
    ORB_CONFIRM_HOUR_COL,
    ORB_DIRECTION_COL,
    ORB_FIRST_BAR_RETURN_COL,
    OpeningRangeBreakStrategy,
    build_per_day_orb,
)

log = structlog.get_logger("run_orb_oos")

LEVERAGE = 5.0  # FIXED.
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
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 5, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date.today())
    p.add_argument(
        "--split",
        type=date.fromisoformat,
        default=date(2025, 12, 1),
        help="train < split <= test (default approximates the mid-point of #17's window)",
    )
    p.add_argument(
        "--min-trailing",
        type=int,
        default=3,
        help="skip days with fewer than N full bars after the 09:00 range bar",
    )
    return p.parse_args(argv)


def _row(s: SetStats) -> str:
    return (
        f"{s.name:<6}{s.n_trades:>8}{s.win_rate * 100:>9.1f}"
        f"{s.mean_signed_move * 100:>13.3f}{s.t_stat:>8.2f}"
        f"{s.gross_return * 100:>11.1f}{s.net_return * 100:>11.1f}"
        f"{s.gross_sharpe:>9.2f}{s.net_sharpe:>9.2f}"
    )


def _verdict(test: SetStats) -> str:
    """Pre-registered decision tree from issue #19."""
    gross_holds = test.t_stat >= 2.0 and test.mean_signed_move > 0
    net_holds = test.net_return > 0 and test.net_sharpe > 0
    if gross_holds and net_holds:
        return (
            "Branch A — NET EDGE HOLDS OOS. Real candidate; cost-validate at "
            "conservative spreads and queue Phase 1.5."
        )
    if test.mean_signed_move > 0:
        return (
            "Branch B — fails OOS, but break-direction CONTINUES on average (mean "
            "signed move > 0). Morning-momentum hint consistent with gap-arbitrage's "
            "daytime echo. File a follow-up issue on entry-bar offsets before "
            "declaring intraday ORB dead."
        )
    return (
        "Branch C — fails OOS AND nothing significant; break-direction MEAN-REVERTS "
        "(fade). Opening-range hypothesis is dead for ^OMX. Next: file an issue for "
        "last-hour drift (Tier-2 source #8)."
    )


def _print_breakdown_counts(name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        print(f"{name}: (empty)")
        return
    direction_counts: Counter[str] = Counter()
    for v in frame[ORB_DIRECTION_COL]:
        if v is Direction.LONG:
            direction_counts["LONG"] += 1
        elif v is Direction.SHORT:
            direction_counts["SHORT"] += 1
        else:
            direction_counts["no-trade"] += 1
    total = sum(direction_counts.values())
    print(
        f"{name:<6}{total:>6} days  "
        f"LONG={direction_counts['LONG']:>3}  SHORT={direction_counts['SHORT']:>3}  "
        f"no-trade={direction_counts['no-trade']:>3}  "
        f"(trade rate {(direction_counts['LONG'] + direction_counts['SHORT']) * 100 / total:.0f}%)"
    )


def _print_hold_length_distribution(frame: pd.DataFrame) -> None:
    holds = [int(v) for v in frame[ORB_BARS_HELD_COL] if int(v) > 0]
    if not holds:
        print("hold length: (no trade days)")
        return
    holds.sort()
    n = len(holds)
    median = holds[n // 2]
    print(
        f"hold length (bars, n={n}):  min={holds[0]}  median={median}  max={holds[-1]}"
    )
    bucket: Counter[int] = Counter(holds)
    for k in sorted(bucket):
        bar = "#" * bucket[k]
        print(f"  {k} bar{'s' if k != 1 else ' '}  {bucket[k]:>3}  {bar}")


def _print_confirm_hour_distribution(frame: pd.DataFrame) -> None:
    # mixed int / None values get coerced to float64 with NaN by pandas, so
    # filter via pd.notna -- `v is not None` would let NaN through.
    rows = [
        (v, d)
        for v, d in zip(frame[ORB_CONFIRM_HOUR_COL], frame[ORB_DIRECTION_COL], strict=True)
        if pd.notna(v)
    ]
    if not rows:
        print("confirm hour: (no trade days)")
        return
    by_hour: Counter[int] = Counter(int(v) for v, _ in rows)
    long_by_hour: Counter[int] = Counter(int(v) for v, d in rows if d is Direction.LONG)
    short_by_hour: Counter[int] = Counter(int(v) for v, d in rows if d is Direction.SHORT)
    print(f"confirm hour (Stockholm local, n={len(rows)}):")
    for h in sorted(by_hour):
        total = by_hour[h]
        bar = "#" * total
        print(
            f"  {h:02d}:00  {total:>3}  (L={long_by_hour[h]:>2}, S={short_by_hour[h]:>2})  {bar}"
        )


def _print_first_bar_return_distribution(frame: pd.DataFrame) -> None:
    vals = [float(v) for v in frame[ORB_FIRST_BAR_RETURN_COL]]
    if not vals:
        print("first-bar return: (empty)")
        return
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    mean = sum(vals_sorted) / n
    median = vals_sorted[n // 2]
    p10 = vals_sorted[max(0, n // 10)]
    p90 = vals_sorted[min(n - 1, (n * 9) // 10)]
    abs_mean = sum(abs(v) for v in vals_sorted) / n
    print(
        f"09:00 bar (close-open)/open, n={n}: "
        f"mean={mean * 100:.3f}%  median={median * 100:.3f}%  "
        f"|mean|={abs_mean * 100:.3f}%  p10={p10 * 100:.3f}%  p90={p90 * 100:.3f}%"
    )
    print(
        "(Reading: a large |mean| would suggest the 09:00 'range' is largely the "
        "overnight gap re-priced — i.e. the strategy is reacting to information "
        "already locked in by 09:00 Stockholm.)"
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    configure_logging()
    args = parse_args(argv)

    cost_pct = float(estimate_round_trip_cost().total_pct)
    breakeven_underlying = cost_pct / LEVERAGE

    bars = get_bars(INSTRUMENT, args.start, args.end, HOURLY_INTERVAL)
    if bars.empty:
        log.error("no_bars", instrument=INSTRUMENT)
        return 1

    synthetic = build_per_day_orb(bars, min_trailing_bars=args.min_trailing)
    if synthetic.empty:
        log.error("no_per_day_rows", instrument=INSTRUMENT)
        return 1

    log.info(
        "orb_oos.start",
        instrument=INSTRUMENT,
        train=f"{synthetic.index.min().date()}..{args.split}",
        test=f"{args.split}..{synthetic.index.max().date()}",
        leverage=LEVERAGE,
        min_trailing=args.min_trailing,
    )

    strategy = OpeningRangeBreakStrategy(INSTRUMENT)
    train, test = run_oos(
        wide_frame=synthetic, strategy=strategy, split=args.split, leverage=LEVERAGE
    )

    print("\n=== OOS validation (opening-range break, ^OMX 1h, 5x, frozen) ===")
    print(
        f"Window: {synthetic.index.min().date()} -> {synthetic.index.max().date()}  "
        f"Split: {args.split}"
    )
    print(
        f"Round-trip cost {cost_pct * 100:.2f}% cert / breakeven "
        f"{breakeven_underlying * 100:.3f}% underlying (per ORB day open->close)\n"
    )
    header = (
        f"{'set':<6}{'trades':>8}{'win%':>9}{'meanMove%':>13}{'t-stat':>8}"
        f"{'gross%':>11}{'net%':>11}{'grSh':>9}{'netSh':>9}"
    )
    print(header)
    print(_row(train))
    print(_row(test))
    print("\nOOS VERDICT (pre-registered tree from #19): " + _verdict(test))

    train_frame = synthetic[synthetic.index < pd.Timestamp(args.split, tz="UTC")]
    test_frame = synthetic[synthetic.index >= pd.Timestamp(args.split, tz="UTC")]

    print("\n=== Break-frequency split ===")
    _print_breakdown_counts("train", train_frame)
    _print_breakdown_counts("test ", test_frame)
    _print_breakdown_counts("all  ", synthetic)

    print("\n=== Hold-length distribution (entry bar -> exit bar, inclusive, full sample) ===")
    _print_hold_length_distribution(synthetic)

    print("\n=== Confirm-hour distribution (Stockholm local, full sample) ===")
    _print_confirm_hour_distribution(synthetic)

    print("\n=== First-bar return distribution (09:00 Stockholm bar, full sample) ===")
    _print_first_bar_return_distribution(synthetic)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
