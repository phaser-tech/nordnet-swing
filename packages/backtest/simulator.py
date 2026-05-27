"""Cert price simulator.

Translates underlying price paths to certificate price paths, correctly
accounting for the daily reset mechanism in leveraged certificates.

KEY INSIGHT: Bull/Bear certificates apply leverage to *daily* returns.
This means holding longer than 1 day introduces volatility decay
(compounding drag). Our strategies exit same-day to avoid this.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd


@dataclass(frozen=True)
class CertSpec:
    """Specification of a leveraged certificate.

    Positive leverage = Bull cert (goes up when underlying goes up).
    Negative leverage = Bear cert (goes up when underlying goes down).

    daily_fee_pct is the daily administration fee charged by the issuer.
    Typical values are 0.0005-0.0010 (i.e. ~0.2-0.4% annualized).
    """

    name: str
    underlying: str
    leverage: Decimal
    daily_fee_pct: Decimal = Decimal("0.0008")

    @property
    def is_bear(self) -> bool:
        return self.leverage < 0

    @property
    def abs_leverage(self) -> Decimal:
        return abs(self.leverage)


def simulate_cert_path(
    underlying_prices: pd.Series,
    cert_spec: CertSpec,
    initial_cert_price: Decimal = Decimal("100"),
) -> pd.Series:
    """Simulate cert price path given an underlying price path.

    Uses the daily reset model:
        cert_t+1 = cert_t * (1 + leverage * underlying_return_t) * (1 - daily_fee)

    Args:
        underlying_prices: Series of underlying prices, indexed by datetime.
        cert_spec: Cert specification (leverage, daily fee).
        initial_cert_price: Starting cert price (arbitrary unit, default 100).

    Returns:
        Series of cert prices with the same index as underlying_prices.
    """
    if len(underlying_prices) == 0:
        return pd.Series(dtype=float, name=cert_spec.name)

    underlying_returns = underlying_prices.pct_change().fillna(0)

    leverage = float(cert_spec.leverage)
    daily_fee = float(cert_spec.daily_fee_pct)

    cert_multipliers = (1 + leverage * underlying_returns) * (1 - daily_fee)

    # Clip multipliers to prevent the cert from going negative (knockout proxy).
    # In reality the cert would be knocked out; here we just clip at near-zero
    # to keep the time series well-defined. Real knockout handling lives in
    # risk/ later.
    cert_multipliers = cert_multipliers.clip(lower=0.001)

    cert_prices = float(initial_cert_price) * cert_multipliers.cumprod()
    cert_prices.name = cert_spec.name

    return cert_prices


def simulate_intraday_trade(
    underlying_entry: Decimal,
    underlying_exit: Decimal,
    cert_spec: CertSpec,
    initial_cert_price: Decimal = Decimal("100"),
) -> tuple[Decimal, Decimal]:
    """Simulate a single intraday trade (entry and exit within same day).

    Within a single day, the daily reset hasn't kicked in, so the cert
    move is linear with the underlying move scaled by leverage. The daily
    fee is also negligible intraday.

    Args:
        underlying_entry: Underlying price at entry.
        underlying_exit: Underlying price at exit (same trading day).
        cert_spec: Cert specification.
        initial_cert_price: Cert price at entry.

    Returns:
        Tuple of (cert_exit_price, cert_return_pct).
    """
    underlying_return = (underlying_exit - underlying_entry) / underlying_entry
    cert_return = cert_spec.leverage * underlying_return
    cert_exit_price = initial_cert_price * (Decimal("1") + cert_return)
    return cert_exit_price, cert_return
