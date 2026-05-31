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
    """Round-trip cost decomposition, as a fraction of the *cert* position.

    `total_pct` is the issuer spread + slippage + (optional) overnight financing
    paid on the cert you actually trade. The default is in cert terms and does
    NOT scale with leverage — a 0.6% cert spread costs 0.6% of the position
    whether the cert is 3x or 15x.

    To express the same cost in *underlying-return* terms — how far the
    underlying must move to cover it — divide by leverage (`in_underlying_terms`),
    since a cert return is `leverage * underlying_return`.

    `overnight_financing_pct` is only non-zero when a strategy explicitly opts
    in via `estimate_round_trip_cost(overnight_nights=N)`. The default (and the
    behaviour for the four open->close Phase-0/1 strategies) is zero.
    """

    courtage_pct: Decimal = Decimal("0")
    spread_pct: Decimal = Decimal("0")
    slippage_pct: Decimal = Decimal("0")
    overnight_financing_pct: Decimal = Decimal("0")

    @property
    def total_pct(self) -> Decimal:
        return (
            self.courtage_pct
            + self.spread_pct
            + self.slippage_pct
            + self.overnight_financing_pct
        )

    def in_underlying_terms(self, leverage: Decimal) -> Decimal:
        """Underlying move needed to cover this cert-terms cost = total / leverage."""
        return self.total_pct / abs(leverage)


@dataclass(frozen=True)
class CostAssumptions:
    """Configurable assumptions about trading costs.

    Defaults are conservative for liquid OMX/Nasdaq Bull/Bear certs (`CERT_PROFILE`).
    Use `FUTURES_PROFILE` for direct index-futures trading instead -- it has a
    radically lower cost wall (~10x lower) because futures are exchange-traded
    with tight bid-ask and small commissions vs issuer-spread certs.

    `overnight_financing_pct_per_night` is the per-night holding cost a
    leveraged certificate charges (issuer borrows `(L-1)x` your capital and
    passes a financing rate through). It is only applied when the caller
    explicitly opts in via `overnight_nights > 0` -- otherwise the rule is
    "no overnight holds" and the rate is irrelevant. Cert default 0.03%/night
    is a conservative current-rate-environment estimate; calibrate against
    real Nordnet Markets statements when we have live data. Futures profile
    sets this to 0 -- futures carry no separate daily-reset financing; the
    cost-of-carry is priced into the futures basis vs spot.
    """

    spread_pct_round_trip: Decimal = Decimal("0.005")  # 0.5% baseline
    spread_widen_stress_multiplier: Decimal = Decimal("3.0")
    slippage_pct_round_trip: Decimal = Decimal("0.001")  # 0.1% baseline
    courtage_pct: Decimal = Decimal("0")  # 0 for Nordnet Markets
    overnight_financing_pct_per_night: Decimal = Decimal("0.0003")  # 0.03%/night


# Pre-baked cost profiles per instrument class.
#
# CERT_PROFILE = the historical Phase-0 default. All five OOS-tested strategies
# (#10/#14/#15/#20/#22) used this. Keeping it equal to `CostAssumptions()`'s
# bare defaults preserves their results bit-for-bit.
CERT_PROFILE = CostAssumptions()

# FUTURES_PROFILE = retail OMXS30 (or equivalent) index-futures trading.
# Source numbers:
#   - OMXS30 tick = 0.125 pts on a ~3400-level index = 12.5 SEK on ~340,000
#     SEK contract notional => one tick ~ 0.0037% of notional. Realistic
#     retail effective spread ~ 1-2 ticks one-way = ~1 bp round-trip after
#     market-impact buffer.
#   - Slippage minimal on liquid futures: ~0.5 bp.
#   - Retail futures broker commission ~ 50-100 SEK per side on OMXS30 =
#     ~5 bp round-trip on a single contract. (Better with volume discounts.)
#   - Overnight financing is 0 in this model: futures don't reset daily;
#     cost-of-carry is in the basis. Backtesting on spot index this is
#     absorbed into price action and not double-counted here.
#   - Stress widening is lower than for certs (~2x vs ~3x) -- exchange order
#     books stay tighter under stress than issuer-quoted certs.
FUTURES_PROFILE = CostAssumptions(
    spread_pct_round_trip=Decimal("0.0001"),  # ~1 bp effective round-trip spread
    spread_widen_stress_multiplier=Decimal("2.0"),
    slippage_pct_round_trip=Decimal("0.00005"),  # ~0.5 bp
    courtage_pct=Decimal("0.0005"),  # ~5 bp retail commission round-trip
    overnight_financing_pct_per_night=Decimal("0"),  # priced into basis
)


def estimate_round_trip_cost(
    *,
    in_stress: bool = False,
    overnight_nights: int = 0,
    assumptions: CostAssumptions | None = None,
) -> CostBreakdown:
    """Estimate round-trip cost as a fraction of the cert position.

    Args:
        in_stress: True during major news releases or high-vol regimes.
                   Applies spread widening multiplier.
        overnight_nights: Number of overnights the trade is held. Default 0
                   matches the project's default no-overnight rule; pass >=1
                   only for explicitly-approved gap-capture strategies
                   (CLAUDE.md "Approved overnight exceptions").
        assumptions: Override default cost assumptions.

    Returns:
        CostBreakdown with courtage, spread, slippage, financing decomposed.
    """
    if assumptions is None:
        assumptions = CostAssumptions()
    if overnight_nights < 0:
        raise ValueError(f"overnight_nights must be >= 0, got {overnight_nights}")

    spread = assumptions.spread_pct_round_trip
    if in_stress:
        spread = spread * assumptions.spread_widen_stress_multiplier

    financing = assumptions.overnight_financing_pct_per_night * Decimal(overnight_nights)

    return CostBreakdown(
        courtage_pct=assumptions.courtage_pct,
        spread_pct=spread,
        slippage_pct=assumptions.slippage_pct_round_trip,
        overnight_financing_pct=financing,
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
    return cost.in_underlying_terms(leverage)


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
