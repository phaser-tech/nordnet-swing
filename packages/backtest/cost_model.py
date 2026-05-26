"""Cost model for Nordnet Markets certificates.

Key facts informing this model:
- Courtage = 0 for Nordnet Markets products via Next API (order > 1000 SEK)
- Main cost is issuer spread (bull/bear certs are market-maker priced, not orderbook)
- Spread can widen 3-5x during major news releases
- No overnight financing if we exit same day
- Daily reset means we don't accumulate compounding drag intraday

Spread assumptions are conservative defaults — actual values should be
calibrated against real Nordnet Markets data once we have access.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class CostBreakdown:
    """Round-trip cost decomposition, in underlying-return terms.

    To convert to cert-return terms, multiply by leverage.
    """

    courtage_pct: Decimal = Decimal("0")
    spread_pct: Decimal = Decimal("0")
    slippage_pct: Decimal = Decimal("0")

    @property
    def total_pct(self) -> Decimal:
        return self.courtage_pct + self.spread_pct + self.slippage_pct

    def in_cert_terms(self, leverage: Decimal) -> Decimal:
        """Cost expressed as a percentage of cert position value."""
        return self.total_pct * abs(leverage)


@dataclass(frozen=True)
class CostAssumptions:
    """Configurable assumptions about cert trading costs.

    Defaults are conservative for liquid OMX/Nasdaq Bull/Bear certs.
    """

    spread_pct_round_trip: Decimal = Decimal("0.005")  # 0.5% baseline
    spread_widen_stress_multiplier: Decimal = Decimal("3.0")
    slippage_pct_round_trip: Decimal = Decimal("0.001")  # 0.1% baseline
    courtage_pct: Decimal = Decimal("0")  # 0 for Nordnet Markets


def estimate_round_trip_cost(
    *,
    in_stress: bool = False,
    assumptions: CostAssumptions | None = None,
) -> CostBreakdown:
    """Estimate round-trip cost in underlying-return terms.

    Args:
        in_stress: True during major news releases or high-vol regimes.
                   Applies spread widening multiplier.
        assumptions: Override default cost assumptions.

    Returns:
        CostBreakdown with courtage, spread, slippage decomposed.
    """
    if assumptions is None:
        assumptions = CostAssumptions()

    spread = assumptions.spread_pct_round_trip
    if in_stress:
        spread = spread * assumptions.spread_widen_stress_multiplier

    return CostBreakdown(
        courtage_pct=assumptions.courtage_pct,
        spread_pct=spread,
        slippage_pct=assumptions.slippage_pct_round_trip,
    )


def required_underlying_move_for_breakeven(
    leverage: Decimal,
    in_stress: bool = False,
    assumptions: CostAssumptions | None = None,
) -> Decimal:
    """How big must the underlying move be to cover round-trip costs?

    This is THE key question for any trade candidate:
    "Does the expected move exceed the cost of attempting?"

    Args:
        leverage: Cert leverage factor (e.g. 5 for 5x bull, -5 for 5x bear).
        in_stress: True during stress regimes.
        assumptions: Override default cost assumptions.

    Returns:
        Required underlying move as a Decimal fraction (e.g. 0.001 = 0.1%).
    """
    cost = estimate_round_trip_cost(in_stress=in_stress, assumptions=assumptions)
    return cost.total_pct / abs(leverage)


def signal_passes_cost_filter(
    *,
    expected_underlying_move_pct: Decimal,
    leverage: Decimal,
    safety_margin: Decimal = Decimal("1.5"),
    in_stress: bool = False,
    assumptions: CostAssumptions | None = None,
) -> bool:
    """Does an expected move comfortably exceed costs?

    Args:
        expected_underlying_move_pct: Expected favorable move in underlying.
        leverage: Cert leverage.
        safety_margin: Multiplier on cost — expected move must exceed
                       this * cost to qualify. 1.5x is a reasonable default.

    Returns:
        True if the signal has enough expected edge to attempt.
    """
    breakeven = required_underlying_move_for_breakeven(
        leverage=leverage,
        in_stress=in_stress,
        assumptions=assumptions,
    )
    return expected_underlying_move_pct >= breakeven * safety_margin
