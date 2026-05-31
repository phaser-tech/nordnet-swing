"""Tests for the cost model."""

from __future__ import annotations

from decimal import Decimal

from packages.backtest.cost_model import (
    CERT_PROFILE,
    FUTURES_PROFILE,
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

    def test_total_pct_is_cert_terms_and_leverage_independent(self) -> None:
        """total_pct is the cert-terms round-trip cost; it does not scale with leverage."""
        cost = estimate_round_trip_cost()
        # CLAUDE.md: issuer spread 0.3-0.8% round-trip on the cert itself.
        assert Decimal("0.003") <= cost.total_pct <= Decimal("0.008")

    def test_in_underlying_terms_divides_by_leverage(self) -> None:
        """Underlying must move cert_cost / leverage to cover round-trip costs."""
        cost = estimate_round_trip_cost()
        assert cost.in_underlying_terms(Decimal("5")) == cost.total_pct / Decimal("5")
        # Bear cert (negative leverage) uses absolute leverage.
        assert cost.in_underlying_terms(Decimal("-5")) == cost.total_pct / Decimal("5")

    def test_cert_and_underlying_costs_are_consistent(self) -> None:
        """The fix: scaling the underlying breakeven by leverage recovers the cert cost.

        Guards against the old leverage^2 inconsistency between the cert-terms
        cost and `required_underlying_move_for_breakeven`.
        """
        cost = estimate_round_trip_cost()
        for lev in (Decimal("3"), Decimal("5"), Decimal("15")):
            assert cost.in_underlying_terms(lev) * lev == cost.total_pct
            assert required_underlying_move_for_breakeven(lev) == cost.in_underlying_terms(lev)


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


class TestOvernightFinancing:
    def test_default_overnight_nights_is_zero_so_existing_callers_unchanged(self) -> None:
        """Backwards compatibility: callers that don't pass overnight_nights see no change."""
        cost = estimate_round_trip_cost()
        assert cost.overnight_financing_pct == Decimal("0")
        assert cost.total_pct == Decimal("0.006")  # same 0.6% as before the extension

    def test_one_night_adds_default_financing(self) -> None:
        cost = estimate_round_trip_cost(overnight_nights=1)
        assert cost.overnight_financing_pct == Decimal("0.0003")
        # 0.5% spread + 0.1% slippage + 0.03% financing = 0.63%
        assert cost.total_pct == Decimal("0.0063")

    def test_financing_scales_linearly_with_nights(self) -> None:
        zero = estimate_round_trip_cost(overnight_nights=0)
        one = estimate_round_trip_cost(overnight_nights=1)
        two = estimate_round_trip_cost(overnight_nights=2)
        assert one.overnight_financing_pct - zero.overnight_financing_pct == Decimal("0.0003")
        assert two.overnight_financing_pct == Decimal("0.0006")

    def test_negative_nights_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="overnight_nights"):
            estimate_round_trip_cost(overnight_nights=-1)

    def test_financing_carries_through_to_underlying_terms(self) -> None:
        """in_underlying_terms divides the *total* (including financing) by leverage."""
        cost = estimate_round_trip_cost(overnight_nights=1)
        assert cost.in_underlying_terms(Decimal("5")) == cost.total_pct / Decimal("5")

    def test_custom_financing_rate(self) -> None:
        higher = CostAssumptions(overnight_financing_pct_per_night=Decimal("0.001"))
        cost = estimate_round_trip_cost(overnight_nights=1, assumptions=higher)
        assert cost.overnight_financing_pct == Decimal("0.001")


class TestCostProfiles:
    """The pre-baked CERT_PROFILE / FUTURES_PROFILE bundles."""

    def test_cert_profile_equals_bare_defaults_regression(self) -> None:
        """CERT_PROFILE MUST equal `CostAssumptions()` so the 5 Phase-0 strategies'
        results are reproducible bit-for-bit when re-run with `assumptions=CERT_PROFILE`."""
        cert = estimate_round_trip_cost(assumptions=CERT_PROFILE)
        bare = estimate_round_trip_cost()
        assert cert == bare

    def test_futures_profile_total_cost_is_about_10x_lower_than_cert(self) -> None:
        cert_total = estimate_round_trip_cost(assumptions=CERT_PROFILE).total_pct
        fut_total = estimate_round_trip_cost(assumptions=FUTURES_PROFILE).total_pct
        ratio = cert_total / fut_total
        assert ratio > Decimal("5"), f"futures should be much cheaper; ratio={ratio}"
        assert ratio < Decimal("20"), f"sanity: futures shouldn't be 20x+ cheaper; ratio={ratio}"

    def test_futures_profile_has_zero_overnight_financing(self) -> None:
        """Futures cost-of-carry is in the basis vs spot, not a separate financing line."""
        cost = estimate_round_trip_cost(overnight_nights=1, assumptions=FUTURES_PROFILE)
        assert cost.overnight_financing_pct == Decimal("0")

    def test_futures_breakeven_under_5_bp_at_5x(self) -> None:
        """Underlying breakeven at 5x exposure under FUTURES_PROFILE must be << cert's 12bp."""
        breakeven = required_underlying_move_for_breakeven(
            leverage=Decimal("5"), assumptions=FUTURES_PROFILE
        )
        # ~6.5 bp cert-terms / 5 = ~1.3 bp underlying. Bound loosely.
        assert breakeven < Decimal("0.0005"), f"got {breakeven} -- futures breakeven too high"

    def test_futures_profile_overnight_call_does_not_add_financing(self) -> None:
        """Sanity: even with overnight_nights=N, futures profile keeps financing at 0."""
        for n in (0, 1, 5, 30):
            cost = estimate_round_trip_cost(
                overnight_nights=n, assumptions=FUTURES_PROFILE
            )
            assert cost.overnight_financing_pct == Decimal("0")
