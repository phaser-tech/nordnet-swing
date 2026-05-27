"""Tests for the cost model."""

from __future__ import annotations

from decimal import Decimal

from packages.backtest.cost_model import (
    CostAssumptions,
    estimate_round_trip_cost,
    required_underlying_move_for_breakeven,
    signal_passes_cost_filter,
)


class TestEstimateRoundTripCost:
    def test_default_costs_have_zero_courtage(self) -> None:
        """Nordnet Markets has zero courtage via Next API."""
        cost = estimate_round_trip_cost()
        assert cost.courtage_pct == Decimal("0")

    def test_normal_regime_total_is_around_60bps(self) -> None:
        """0.5% spread + 0.1% slippage = 0.6% total in normal regime."""
        cost = estimate_round_trip_cost(in_stress=False)
        assert cost.total_pct == Decimal("0.006")

    def test_stress_regime_widens_spread(self) -> None:
        """In stress, spread widens by the multiplier (default 3x)."""
        normal = estimate_round_trip_cost(in_stress=False)
        stress = estimate_round_trip_cost(in_stress=True)
        # Spread becomes 0.5% * 3 = 1.5%, plus unchanged 0.1% slippage = 1.6%
        assert stress.total_pct == Decimal("0.016")
        assert stress.spread_pct == normal.spread_pct * Decimal("3.0")

    def test_in_cert_terms_scales_by_leverage(self) -> None:
        """Cost in cert terms = cost in underlying terms * leverage."""
        cost = estimate_round_trip_cost()
        assert cost.in_cert_terms(Decimal("5")) == cost.total_pct * Decimal("5")
        # Same for negative leverage (bear cert) — uses abs.
        assert cost.in_cert_terms(Decimal("-5")) == cost.total_pct * Decimal("5")


class TestBreakeven:
    def test_5x_leverage_needs_smaller_underlying_move(self) -> None:
        """Higher leverage = smaller required underlying move."""
        move_5x = required_underlying_move_for_breakeven(Decimal("5"))
        move_10x = required_underlying_move_for_breakeven(Decimal("10"))
        assert move_10x < move_5x

    def test_breakeven_5x_normal_regime(self) -> None:
        """At 5x leverage, normal regime, need ~0.12% underlying move to break even."""
        move = required_underlying_move_for_breakeven(Decimal("5"))
        # 0.6% / 5 = 0.12%
        assert move == Decimal("0.006") / Decimal("5")

    def test_breakeven_symmetric_in_bear_cert(self) -> None:
        """Bear cert (negative leverage) has same breakeven by absolute leverage."""
        bull_move = required_underlying_move_for_breakeven(Decimal("5"))
        bear_move = required_underlying_move_for_breakeven(Decimal("-5"))
        assert bull_move == bear_move


class TestCostFilter:
    def test_signal_with_small_move_blocked(self) -> None:
        """Tiny expected move doesn't pass the filter."""
        passes = signal_passes_cost_filter(
            expected_underlying_move_pct=Decimal("0.0001"),  # 0.01%
            leverage=Decimal("5"),
        )
        assert not passes

    def test_signal_with_large_move_passes(self) -> None:
        """A clearly profitable expected move passes."""
        passes = signal_passes_cost_filter(
            expected_underlying_move_pct=Decimal("0.01"),  # 1%
            leverage=Decimal("5"),
        )
        assert passes

    def test_safety_margin_filters_marginal_signals(self) -> None:
        """A signal at exactly breakeven shouldn't pass with default 1.5x margin."""
        breakeven = required_underlying_move_for_breakeven(Decimal("5"))
        # At breakeven exactly — should fail with safety margin > 1
        passes = signal_passes_cost_filter(
            expected_underlying_move_pct=breakeven,
            leverage=Decimal("5"),
            safety_margin=Decimal("1.5"),
        )
        assert not passes

        # At 1.5x breakeven — should pass
        passes_safe = signal_passes_cost_filter(
            expected_underlying_move_pct=breakeven * Decimal("1.5"),
            leverage=Decimal("5"),
            safety_margin=Decimal("1.5"),
        )
        assert passes_safe

    def test_stress_regime_blocks_signals_normal_would_pass(self) -> None:
        """Same signal can pass normal but fail in stress."""
        expected_move = Decimal("0.003")  # 0.3% underlying
        leverage = Decimal("5")

        passes_normal = signal_passes_cost_filter(
            expected_underlying_move_pct=expected_move,
            leverage=leverage,
            in_stress=False,
        )
        passes_stress = signal_passes_cost_filter(
            expected_underlying_move_pct=expected_move,
            leverage=leverage,
            in_stress=True,
        )

        assert passes_normal
        assert not passes_stress


class TestCustomAssumptions:
    def test_can_override_assumptions(self) -> None:
        """Custom CostAssumptions override defaults."""
        wide_spread = CostAssumptions(spread_pct_round_trip=Decimal("0.02"))
        cost = estimate_round_trip_cost(assumptions=wide_spread)
        assert cost.spread_pct == Decimal("0.02")
