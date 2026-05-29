"""Unit + property tests for the bounce decomposition. Fixtures only."""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from packages.backtest.reversion_decomposition import (
    decompose,
    next_day_legs,
)
from packages.core.domain.signal import Direction

# 5 bars -> next_day_legs yields 4 trigger rows (the last bar has no T+1).
_OPEN = [100.0, 101.0, 98.0, 103.0, 100.0]
_CLOSE = [100.0, 110.0, 95.0, 108.0, 100.0]


def _frame(opens: list[float], closes: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range("2022-11-01", periods=len(closes), tz="UTC")
    return pd.DataFrame(
        {
            "Open": opens,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1000] * len(closes),
        },
        index=pd.DatetimeIndex(idx, name="ts"),
    )


class TestNextDayLegs:
    def test_drops_last_bar(self) -> None:
        df = _frame(_OPEN, _CLOSE)
        legs = next_day_legs(df)
        assert len(legs) == len(df) - 1
        assert df.index[-1] not in legs.index

    def test_leg_values_for_first_trigger(self) -> None:
        df = _frame(_OPEN, _CLOSE)
        legs = next_day_legs(df)
        first = legs.iloc[0]
        assert first["gap"] == pytest.approx(101.0 / 100.0 - 1.0)
        assert first["intraday"] == pytest.approx(110.0 / 101.0 - 1.0)
        assert first["full_day"] == pytest.approx(110.0 / 100.0 - 1.0)

    def test_compounding_identity_per_row(self) -> None:
        df = _frame(_OPEN, _CLOSE)
        legs = next_day_legs(df)
        recomposed = (1.0 + legs["gap"]) * (1.0 + legs["intraday"]) - 1.0
        pd.testing.assert_series_equal(
            recomposed, legs["full_day"], check_names=False
        )


def _bet(df: pd.DataFrame, directions: list[Direction | None]) -> pd.Series:
    """A bet series aligned to df.index (pad to full length with None)."""
    padded = directions + [None] * (len(df) - len(directions))
    return pd.Series(padded, index=df.index, dtype=object)


class TestDecompose:
    def test_buckets_select_by_direction(self) -> None:
        df = _frame(_OPEN, _CLOSE)
        # T0 LONG, T1 SHORT, T2 LONG, T3 none (T4 dropped anyway)
        bet = _bet(df, [Direction.LONG, Direction.SHORT, Direction.LONG, None])
        out = decompose(df, bet)

        assert out["down-extreme"].n_events == 2  # T0, T2
        assert out["up-extreme"].n_events == 1  # T1
        assert out["down-extreme"].bet_direction == Direction.LONG
        assert out["up-extreme"].bet_direction == Direction.SHORT

    def test_down_extreme_reversion_means_positive_here(self) -> None:
        df = _frame(_OPEN, _CLOSE)
        bet = _bet(df, [Direction.LONG, Direction.SHORT, Direction.LONG, None])
        down = decompose(df, bet)["down-extreme"]
        # T0 & T2 both rose intraday and gapped up -> reversion (LONG) is positive
        assert down.gap.reversion_mean > 0
        assert down.intraday.reversion_mean > 0
        assert down.intraday.reversion_hit_rate == pytest.approx(1.0)

    def test_short_bucket_sign_flips_into_reversion(self) -> None:
        df = _frame(_OPEN, _CLOSE)
        bet = _bet(df, [Direction.LONG, Direction.SHORT, Direction.LONG, None])
        up = decompose(df, bet)["up-extreme"]
        # T1 fell on both legs; for a SHORT bet that is reversion -> signed positive
        assert up.gap.mean < 0
        assert up.gap.reversion_mean == pytest.approx(-up.gap.mean)
        assert up.intraday.reversion_mean > 0
        # single event -> t-stat undefined, reported as 0
        assert up.intraday.t_stat == 0.0

    def test_empty_bucket_is_well_defined(self) -> None:
        df = _frame(_OPEN, _CLOSE)
        bet = _bet(df, [Direction.LONG, Direction.LONG, Direction.LONG, None])
        up = decompose(df, bet)["up-extreme"]
        assert up.n_events == 0
        assert up.gap.n == 0
        assert up.gap.mean == 0.0
        assert up.intraday.reversion_hit_rate == 0.0


_price = st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)


class TestIdentityProperty:
    @given(
        opens=st.lists(_price, min_size=3, max_size=40),
        closes=st.lists(_price, min_size=3, max_size=40),
    )
    def test_gap_times_intraday_equals_full_day(
        self, opens: list[float], closes: list[float]
    ) -> None:
        n = min(len(opens), len(closes))
        df = _frame(opens[:n], closes[:n])
        legs = next_day_legs(df)
        recomposed = (1.0 + legs["gap"]) * (1.0 + legs["intraday"]) - 1.0
        for got, expected in zip(recomposed, legs["full_day"], strict=True):
            assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)
