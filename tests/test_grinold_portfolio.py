"""Unit tests for the Grinold-portfolio strategy. Fixtures only."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from packages.core.domain.signal import Direction
from packages.strategies.grinold_portfolio import (
    GRINOLD_PREDICTION_COL,
    FittedCoefficients,
    GrinoldPortfolioStrategy,
    attach_prediction,
    build_signal_columns,
    fit_ols,
    predict_series,
)


def _wide_frame_with_two_signal_tickers() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
        ),
        name="ts",
    ).tz_localize("UTC")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1000] * 5,
            "^GSPC_Close": [4000.0, 4040.0, 4040.0, 4000.0, 4040.0],
            "^VIX_Close": [20.0, 21.0, 22.0, 21.0, 20.0],
        },
        index=idx,
    )


def _two_signal_specs() -> dict[str, tuple[str, str]]:
    return {"spx_2d": ("^GSPC", "pct2"), "vix_2d": ("^VIX", "diff2")}


class TestBuildSignalColumns:
    def test_pct2_and_diff2_produce_shifted_series(self) -> None:
        wide = _wide_frame_with_two_signal_tickers()
        cols = build_signal_columns(wide, _two_signal_specs())
        assert list(cols.columns) == ["spx_2d", "vix_2d"]
        # pct_change(2) at i=2: (4040-4000)/4000 = +0.01. .shift(1) -> at i=3.
        assert cols["spx_2d"].iloc[3] == pytest.approx(0.01)
        # diff(2) at i=2: 22-20=2. .shift(1) -> at i=3.
        assert cols["vix_2d"].iloc[3] == pytest.approx(2.0)

    def test_first_rows_are_nan_due_to_lookback_and_shift(self) -> None:
        wide = _wide_frame_with_two_signal_tickers()
        cols = build_signal_columns(wide, _two_signal_specs())
        # 2 periods of pct_change/diff + 1 shift = 3 leading NaN rows.
        for i in range(3):
            assert pd.isna(cols["spx_2d"].iloc[i])
            assert pd.isna(cols["vix_2d"].iloc[i])

    def test_missing_ticker_column_raises(self) -> None:
        wide = _wide_frame_with_two_signal_tickers().drop(columns=["^VIX_Close"])
        with pytest.raises(ValueError, match="missing column"):
            build_signal_columns(wide, _two_signal_specs())


class TestFitOls:
    def test_recovers_known_linear_combination(self) -> None:
        """Construct y = 0.1 + 0.5 * x1 - 0.3 * x2 and assert the fit returns
        those coefficients within numerical precision."""
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        y = 0.1 + 0.5 * x1 - 0.3 * x2
        idx = pd.RangeIndex(n)
        signals = pd.DataFrame({"a": x1, "b": x2}, index=idx)
        omx = pd.Series(y, index=idx)
        mask = pd.Series([True] * n, index=idx)
        coeffs = fit_ols(signals, omx, mask)
        assert coeffs.signal_names == ("a", "b")
        assert coeffs.intercept == pytest.approx(0.1, abs=1e-10)
        assert coeffs.betas[0] == pytest.approx(0.5, abs=1e-10)
        assert coeffs.betas[1] == pytest.approx(-0.3, abs=1e-10)

    def test_only_train_rows_used(self) -> None:
        """A coefficient fit only on train rows should ignore test-row contamination."""
        n_train, n_test = 100, 100
        idx = pd.RangeIndex(n_train + n_test)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n_train + n_test)
        # train: y = 0.5x.   test rows: y = -10x  (would corrupt the fit if used)
        y = np.concatenate([0.5 * x[:n_train], -10.0 * x[n_train:]])
        mask = pd.Series([True] * n_train + [False] * n_test, index=idx)
        signals = pd.DataFrame({"a": x}, index=idx)
        coeffs = fit_ols(signals, pd.Series(y, index=idx), mask)
        assert coeffs.betas[0] == pytest.approx(0.5, abs=1e-10)


class TestPredictSeries:
    def test_applies_intercept_plus_betas(self) -> None:
        idx = pd.RangeIndex(3)
        signals = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}, index=idx)
        coeffs = FittedCoefficients(
            signal_names=("a", "b"), intercept=0.5, betas=(2.0, -0.1)
        )
        pred = predict_series(signals, coeffs)
        # row 0: 0.5 + 2*1 + (-0.1)*10 = 0.5 + 2 - 1 = 1.5
        assert pred.iloc[0] == pytest.approx(1.5)
        assert pred.iloc[1] == pytest.approx(0.5 + 2 * 2 - 0.1 * 20)
        assert pred.name == GRINOLD_PREDICTION_COL

    def test_nan_signal_yields_nan_prediction(self) -> None:
        idx = pd.RangeIndex(2)
        signals = pd.DataFrame({"a": [1.0, np.nan]}, index=idx)
        coeffs = FittedCoefficients(signal_names=("a",), intercept=0.0, betas=(1.0,))
        pred = predict_series(signals, coeffs)
        assert pred.iloc[0] == pytest.approx(1.0)
        assert pd.isna(pred.iloc[1])


class TestAttachPrediction:
    def test_attaches_aligned_column(self) -> None:
        wide = _wide_frame_with_two_signal_tickers()
        pred = pd.Series([0.001, 0.002, 0.003, 0.004, 0.005], index=wide.index)
        out = attach_prediction(wide, pred)
        assert GRINOLD_PREDICTION_COL in out.columns
        assert out[GRINOLD_PREDICTION_COL].iloc[2] == pytest.approx(0.003)


class TestStrategyEmitter:
    def _frame_with_predictions(self, preds: list[float]) -> pd.DataFrame:
        idx = pd.DatetimeIndex(
            pd.to_datetime([f"2024-01-{2 + i:02d}" for i in range(len(preds))]),
            name="ts",
        ).tz_localize("UTC")
        return pd.DataFrame(
            {
                "Open": [100.0 + i for i in range(len(preds))],
                "High": [101.0 + i for i in range(len(preds))],
                "Low": [99.0 + i for i in range(len(preds))],
                "Close": [100.5 + i for i in range(len(preds))],
                "Volume": [1000] * len(preds),
                GRINOLD_PREDICTION_COL: preds,
            },
            index=idx,
        )

    def test_emits_long_when_pred_above_threshold(self) -> None:
        wide = self._frame_with_predictions([0.0, 0.005, -0.002])
        sigs = list(GrinoldPortfolioStrategy("TEST", threshold=0.003).generate_signals(wide))
        assert len(sigs) == 1
        assert sigs[0].direction == Direction.LONG
        assert sigs[0].suggested_entry == Decimal("101.0")  # Open at the signal row

    def test_emits_short_when_pred_below_negative_threshold(self) -> None:
        wide = self._frame_with_predictions([0.0, -0.005])
        sigs = list(GrinoldPortfolioStrategy("TEST", threshold=0.003).generate_signals(wide))
        assert len(sigs) == 1
        assert sigs[0].direction == Direction.SHORT

    def test_skips_below_threshold(self) -> None:
        wide = self._frame_with_predictions([0.002, -0.002, 0.001])
        sigs = list(GrinoldPortfolioStrategy("TEST", threshold=0.003).generate_signals(wide))
        assert sigs == []

    def test_skips_nan_prediction(self) -> None:
        wide = self._frame_with_predictions([float("nan"), 0.005])
        sigs = list(GrinoldPortfolioStrategy("TEST", threshold=0.003).generate_signals(wide))
        assert len(sigs) == 1  # NaN row skipped, second row emits

    def test_rejects_frame_without_prediction_column(self) -> None:
        wide = _wide_frame_with_two_signal_tickers()
        with pytest.raises(ValueError, match=GRINOLD_PREDICTION_COL):
            list(GrinoldPortfolioStrategy("TEST", threshold=0.001).generate_signals(wide))

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_nonpositive_threshold(self, bad: float) -> None:
        with pytest.raises(ValueError, match="threshold"):
            GrinoldPortfolioStrategy("TEST", threshold=bad)


class TestNoLookahead:
    def test_strategy_is_pure_over_precomputed_column(self) -> None:
        """The strategy's decision for any row depends ONLY on that row's
        prediction value -- mutating other rows' predictions doesn't change
        which signals fire at the unchanged row."""
        base = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000] * 3,
                GRINOLD_PREDICTION_COL: [0.005, 0.0, -0.005],
            },
            index=pd.DatetimeIndex(
                pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                name="ts",
            ).tz_localize("UTC"),
        )
        strat = GrinoldPortfolioStrategy("TEST", threshold=0.003)
        baseline = {(s.timestamp, s.direction) for s in strat.generate_signals(base)}

        mutated = base.copy()
        mutated.iloc[1, mutated.columns.get_loc(GRINOLD_PREDICTION_COL)] = 99.0
        after = {(s.timestamp, s.direction) for s in strat.generate_signals(mutated)}
        # The middle row's prediction was below threshold (0.0); now far above.
        # The OUTER rows' decisions must be unchanged.
        for ts_dir in baseline:
            assert ts_dir in after
