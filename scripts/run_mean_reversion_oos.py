"""CLI: OOS validation + bounce decomposition for mean-reversion-after-extreme.

Composition root. Loads OMX daily bars, runs the FROZEN strategy over train
(< split) and test (>= split) through the shared `oos.py` harness, then runs the
diagnostic decomposition of the T+1 bounce (gap / intraday / full-day, bucketed
by extreme direction).

The OOS verdict scores the *intraday* (open->close) leg of T+1 — the only leg
tradeable under the no-overnight rule. The decomposition shows where the
reversion actually lives, so we can tell a real intraday edge apart from an
untradeable overnight gap artifact.

Fixed 5x leverage, default cost — apples-to-apples with the cross-asset OOS run.
No tuning on the test set.

Usage:
    uv run python scripts/run_mean_reversion_oos.py
    uv run python scripts/run_mean_reversion_oos.py --split 2023-01-01 --sigma 2.0
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
from packages.backtest.reversion_decomposition import (
    BucketDecomposition,
    LegStats,
    decompose,
)
from packages.market_data.historical import get_bars
from packages.strategies.mean_reversion_extreme import (
    MeanReversionExtremeStrategy,
    bet_direction_series,
)

log = structlog.get_logger("run_mean_reversion_oos")

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
    p.add_argument("--window", type=int, default=60, help="trailing z-score window (days)")
    p.add_argument("--sigma", type=float, default=2.0, help="extreme threshold in sigmas")
    return p.parse_args(argv)


def _oos_row(s: SetStats) -> str:
    return (
        f"{s.name:<6}{s.n_trades:>8}{s.win_rate * 100:>9.1f}"
        f"{s.mean_signed_move * 100:>13.3f}{s.t_stat:>8.2f}"
        f"{s.gross_return * 100:>11.1f}{s.net_return * 100:>11.1f}"
        f"{s.gross_sharpe:>9.2f}{s.net_sharpe:>9.2f}"
    )


def _oos_verdict(test: SetStats) -> str:
    gross_holds = test.t_stat >= 2.0 and test.mean_signed_move > 0
    net_holds = test.net_return > 0 and test.net_sharpe > 0
    if gross_holds and net_holds:
        return "NET EDGE HOLDS OOS — intraday reversion is significant AND net-positive."
    if gross_holds and not net_holds:
        return (
            "GROSS intraday reversion survives OOS but NET does not — costs eat it. "
            "Not tradeable at 5x as-is."
        )
    return "NO EDGE OOS — the intraday reversion leg does not survive out-of-sample."


def _leg_row(label: str, leg: LegStats, bet_sign: float) -> str:
    rev_t = leg.t_stat * bet_sign  # signed so +ve t => reverting
    return (
        f"  {label:<10}{leg.n:>6}{leg.mean * 100:>11.3f}"
        f"{leg.reversion_mean * 100:>13.3f}{rev_t:>8.2f}{leg.reversion_hit_rate * 100:>9.1f}"
    )


def _reverts(leg: LegStats) -> bool:
    """A leg shows reversion if its bet-signed mean is positive AND significant."""
    return leg.reversion_mean > 0 and abs(leg.t_stat) >= 2.0


def _decomp_verdict(buckets: dict[str, BucketDecomposition]) -> str:
    any_intraday = any(_reverts(b.intraday) for b in buckets.values())
    any_gap = any(_reverts(b.gap) for b in buckets.values())
    if any_intraday:
        return (
            "Intraday reversion is present and significant — a real (tradeable) candidate. "
            "Cross-check the NET OOS verdict above for whether it survives cost at 5x."
        )
    if any_gap:
        return (
            "Reversion lives in the GAP while the intraday leg is flat — and the gap is "
            "untradeable under the no-overnight rule. Gap-arbitrage pattern confirmed for "
            "reaction signals: the 'edge' is an overnight artifact, not a tradeable move."
        )
    return "No significant reversion in either the gap or the intraday leg at this threshold."


def _print_decomposition(buckets: dict[str, BucketDecomposition]) -> None:
    print("\n=== Bounce decomposition (T+1, full sample, raw underlying returns) ===")
    print("rawMean% = mean leg return as observed; revMean%/revT = signed by the bet "
          "(+ve => reverting); hit% = share moving in the reversion direction.\n")
    header = f"  {'leg':<10}{'n':>6}{'rawMean%':>11}{'revMean%':>13}{'revT':>8}{'hit%':>9}"
    for bucket in buckets.values():
        sign = 1.0 if bucket.bet_direction.value == "long" else -1.0
        print(f"{bucket.name}  (n={bucket.n_events}, bet {bucket.bet_direction.value.upper()} on T+1)")
        print(header)
        print(_leg_row("gap", bucket.gap, sign))
        print(_leg_row("intraday", bucket.intraday, sign))
        print(_leg_row("full_day", bucket.full_day, sign))
        print()
    print("DECOMPOSITION VERDICT: " + _decomp_verdict(buckets) + "\n")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    configure_logging()
    args = parse_args(argv)

    cost_pct = float(estimate_round_trip_cost().total_pct)
    breakeven_underlying = cost_pct / LEVERAGE

    bars = get_bars(args.instrument, args.start, args.end, "1d")
    if bars.empty:
        log.error("no_bars", instrument=args.instrument)
        return 1

    # Frozen config: window/sigma fixed for the whole run, no tuning on the test set.
    strategy = MeanReversionExtremeStrategy(
        args.instrument, window=args.window, sigma=args.sigma
    )

    log.info(
        "mean_reversion_oos.start",
        instrument=args.instrument,
        train=f"{args.start}..{args.split}",
        test=f"{args.split}..{args.end}",
        window=args.window,
        sigma=args.sigma,
        leverage=LEVERAGE,
    )
    train, test = run_oos(
        wide_frame=bars, strategy=strategy, split=args.split, leverage=LEVERAGE
    )

    print(
        f"\n=== OOS validation (mean reversion after >={args.sigma:g}sigma move, "
        f"5x, frozen) ==="
    )
    print(f"Train: {args.start} -> {args.split}    Test: {args.split} -> {args.end}")
    print(f"z-window {args.window}d   sigma {args.sigma:g}")
    print(
        f"Round-trip cost {cost_pct * 100:.2f}% cert / breakeven "
        f"{breakeven_underlying * 100:.3f}% underlying (intraday T+1 leg)\n"
    )
    header = (
        f"{'set':<6}{'trades':>8}{'win%':>9}{'meanMove%':>13}{'t-stat':>8}"
        f"{'gross%':>11}{'net%':>11}{'grSh':>9}{'netSh':>9}"
    )
    print(header)
    print(_oos_row(train))
    print(_oos_row(test))
    print("\nOOS VERDICT (tradeable intraday leg): " + _oos_verdict(test))

    bet = bet_direction_series(bars, window=args.window, sigma=args.sigma)
    _print_decomposition(decompose(bars, bet))
    return 0


if __name__ == "__main__":
    sys.exit(main())
