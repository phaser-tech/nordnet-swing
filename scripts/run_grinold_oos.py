"""CLI: OOS validation for the Grinold-portfolio strategy at open->close.

Implements Rank 2 from the edge-frontier memo (PR #23): combine 11 weakly-
predictive cross-asset signals via OLS fitted on train only, applied blind to
test. Tests whether IR = IC * sqrt(breadth) lifts the (failed-individually)
open->close edge above the cost wall.

Pipeline (every step lookahead-safe):
  1. Load OMX + each signal ticker's daily bars (2018-today).
  2. Build per-signal .shift(1) series (so decisions at Stockholm open T use
     only data <= T-1 close).
  3. Split rows at --split (default 2023-01-01).
  4. Fit OLS on TRAIN rows only -> frozen coefficients.
  5. Predict on full sample using the frozen coefficients.
  6. Threshold = THRESHOLD_SIGMA_MULT * std(train_predictions). Pre-registered
     on train only.
  7. Run the OOS harness with --cost-profile {cert,futures}.

Same OOS discipline as every other strategy on this codebase: zero tuning on
test, gross AND net reported, both profiles available.

Usage:
    uv run python scripts/run_grinold_oos.py                  # default cert
    uv run python scripts/run_grinold_oos.py --cost-profile futures
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import structlog
from dotenv import load_dotenv
from scipy import stats  # type: ignore[import-untyped]

from packages.backtest.cost_model import (
    CERT_PROFILE,
    FUTURES_PROFILE,
    CostAssumptions,
    estimate_round_trip_cost,
)
from packages.backtest.oos import SetStats, run_oos
from packages.market_data.historical import get_bars
from packages.strategies.grinold_portfolio import (
    GRINOLD_PREDICTION_COL,
    SIGNAL_SPECS,
    GrinoldPortfolioStrategy,
    attach_prediction,
    build_signal_columns,
    fit_ols,
    predict_series,
)

log = structlog.get_logger("run_grinold_oos")

LEVERAGE = 5.0
INSTRUMENT = "^OMX"
THRESHOLD_SIGMA_MULT = 0.5  # PRE-REGISTERED: threshold = 0.5 * std(train_pred)

_COST_PROFILES: dict[str, CostAssumptions] = {
    "cert": CERT_PROFILE,
    "futures": FUTURES_PROFILE,
}


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
    p.add_argument(
        "--cost-profile",
        choices=sorted(_COST_PROFILES),
        default="cert",
        help="Instrument cost profile (cert is the historical default; futures is ~10x cheaper).",
    )
    return p.parse_args(argv)


def _load_wide_frame(start: date, end: date) -> pd.DataFrame:
    omx = get_bars(INSTRUMENT, start, end, "1d")
    if omx.empty:
        raise RuntimeError(f"no bars for {INSTRUMENT}")
    wide = omx[["Open", "High", "Low", "Close", "Volume"]].copy()
    for _, (ticker, _) in SIGNAL_SPECS.items():
        bars = get_bars(ticker, start, end, "1d")
        col = f"{ticker}_Close"
        wide[col] = bars["Close"].reindex(omx.index).ffill() if not bars.empty else float("nan")
    return wide


def _per_signal_train_tstats(
    signal_cols: pd.DataFrame, omx_ret: pd.Series, train_mask: pd.Series
) -> list[tuple[str, float, float, float]]:
    """Univariate Pearson r + p + signed t for each signal on train rows."""
    df = signal_cols.copy()
    df["__y"] = omx_ret
    df = df.loc[train_mask].dropna()
    out: list[tuple[str, float, float, float]] = []
    for name in signal_cols.columns:
        x = df[name].to_numpy()
        y = df["__y"].to_numpy()
        if x.std() == 0:
            continue
        r, p = stats.pearsonr(x, y)
        n = len(x)
        t = r * math.sqrt(n - 2) / math.sqrt(max(1e-15, 1 - r * r))
        out.append((name, float(r), float(p), float(t)))
    out.sort(key=lambda x: abs(x[3]), reverse=True)
    return out


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
    if gross_holds and net_holds:
        return (
            "NET EDGE HOLDS OOS — Grinold breadth lift is significant AND net-positive. "
            "Cost-validate at conservative spreads and queue Phase 1.5."
        )
    if net_holds:
        return (
            "Net is POSITIVE but the signed-move t-stat is < 2.0 in test. "
            "Soft positive -- worth a sensitivity check (threshold, signal pool) "
            "before drawing strong conclusions."
        )
    if test.mean_signed_move > 0:
        return (
            "Net negative but mean signed move is positive -- the model points the "
            "right way but the prediction magnitude is below cost. Tightening the "
            "threshold or adding orthogonalised signals is the natural next test."
        )
    return (
        "Grinold breadth construction does not help here -- the model's predictions "
        "have the wrong sign in test. Either the train signals don't generalise, "
        "or the regime shifted. Consistent with the EU/US analysis branch finding "
        "that 2024-26 is the strongest single regime but train is dominated by 2018-22."
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    configure_logging()
    args = parse_args(argv)

    assumptions = _COST_PROFILES[args.cost_profile]
    cost_breakdown = estimate_round_trip_cost(assumptions=assumptions)
    cost_pct = float(cost_breakdown.total_pct)
    breakeven_underlying = cost_pct / LEVERAGE

    wide = _load_wide_frame(args.start, args.end)
    if wide.empty:
        log.error("no_wide_frame")
        return 1

    signal_cols = build_signal_columns(wide)
    omx_ret = (wide["Close"] - wide["Open"]) / wide["Open"]
    split_ts = pd.Timestamp(args.split, tz=wide.index.tz)
    train_mask = pd.Series(wide.index < split_ts, index=wide.index)

    coeffs = fit_ols(signal_cols, omx_ret, train_mask)
    pred = predict_series(signal_cols, coeffs)
    train_pred_std = float(pred.loc[train_mask].dropna().std())
    threshold = THRESHOLD_SIGMA_MULT * train_pred_std

    wide_with_pred = attach_prediction(wide, pred)
    # Drop rows with NaN prediction (early rows + any data gap) so oos.py
    # iterates only valid rows.
    wide_with_pred = wide_with_pred.dropna(subset=[GRINOLD_PREDICTION_COL])
    if wide_with_pred.empty:
        log.error("no_valid_rows")
        return 1

    strategy = GrinoldPortfolioStrategy(INSTRUMENT, threshold=threshold)
    train_stats, test_stats = run_oos(
        wide_frame=wide_with_pred,
        strategy=strategy,
        split=args.split,
        leverage=LEVERAGE,
        cost_pct=cost_pct,
    )

    print(
        f"\n=== OOS validation (Grinold portfolio, ^OMX open->close, 5x, "
        f"profile={args.cost_profile.upper()}) ==="
    )
    print(
        f"Window: {wide.index.min().date()} -> {wide.index.max().date()}  "
        f"Split: {args.split}"
    )
    print(
        f"Round-trip cost {cost_pct * 100:.3f}% / breakeven {breakeven_underlying * 100:.4f}% "
        f"underlying per trade"
    )
    print(
        f"Threshold: {THRESHOLD_SIGMA_MULT} * std(train_pred) = "
        f"{threshold * 100:.4f}%   (train_pred_std = {train_pred_std * 100:.4f}%)"
    )

    print("\n=== Train per-signal univariate (Pearson r, p, t -- ranked by |t|) ===")
    print(f"  {'signal':<12}  {'r':>8}  {'p':>7}  {'t':>7}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*7}  {'-'*7}")
    for name, r, p, t in _per_signal_train_tstats(signal_cols, omx_ret, train_mask):
        star = " *" if p < 0.05 else "  "
        print(f"  {name:<12}  {r:>+8.4f}  {p:>6.3f}{star}  {t:>+7.2f}")

    print("\n=== Frozen OLS coefficients (fit on train only) ===")
    print(f"  intercept   {coeffs.intercept * 100:>+11.5f}%")
    for name, beta in zip(coeffs.signal_names, coeffs.betas, strict=True):
        print(f"  {name:<12} {beta:>+12.5f}")

    print("\n=== OOS gross/net ===")
    header = (
        f"{'set':<6}{'trades':>8}{'win%':>9}{'meanMove%':>13}{'t-stat':>8}"
        f"{'gross%':>11}{'net%':>11}{'grSh':>9}{'netSh':>9}"
    )
    print(header)
    print(_row(train_stats))
    print(_row(test_stats))

    # Trade rates per set
    train_rows = wide_with_pred[wide_with_pred.index < split_ts]
    test_rows = wide_with_pred[wide_with_pred.index >= split_ts]
    train_rate = train_stats.n_trades * 100 / len(train_rows) if len(train_rows) else 0.0
    test_rate = test_stats.n_trades * 100 / len(test_rows) if len(test_rows) else 0.0
    print(
        f"\nTrade rates: train {train_rate:.1f}% ({train_stats.n_trades}/{len(train_rows)}), "
        f"test {test_rate:.1f}% ({test_stats.n_trades}/{len(test_rows)})"
    )
    print("\nVERDICT: " + _verdict(test_stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
