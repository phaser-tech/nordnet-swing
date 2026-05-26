"""Tests for the cert price simulator."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from packages.backtest.simulator import (
    CertSpec,
    simulate_cert_path,
    simulate_intraday_trade,
)


@pytest.fixture
def bull_5x() -> CertSpec:
    return CertSpec(
        name="BULL OMX X5 N",
        underlying="^OMX",
        leverage=Decimal("5"),
    )


@pytest.fixture
def bear_5x() -> CertSpec:
    return CertSpec(
        name="BEAR OMX X5 N",
        underlying="^OMX",
        leverage=Decimal("-5"),
    )


class TestCertSpec:
    def test_bull_cert_is_not_bear(self, bull_5x: CertSpec) -> None:
        assert not bull_5x.is_bear

    def test_bear_cert_is_bear(self, bear_5x: CertSpec) -> None:
        assert bear_5x.is_bear

    def test_abs_leverage_normalizes_negative(self, bear_5x: CertSpec) -> None:
        assert bear_5x.abs_leverage == Decimal("5")


class TestIntradayTrade:
    def test_bull_cert_gains_5x_on_underlying_up(self, bull_5x: CertSpec) -> None:
        """1% up in underlying = 5% up in 5x bull cert (intraday, no fee impact)."""
        _, ret = simulate_intraday_trade(
            underlying_entry=Decimal("1000"),
            underlying_exit=Decimal("1010"),
            cert_spec=bull_5x,
        )
        assert ret == Decimal("0.05")

    def test_bear_cert_gains_when_underlying_down(self, bear_5x: CertSpec) -> None:
        """1% down in underlying = 5% up in 5x bear cert."""
        _, ret = simulate_intraday_trade(
            underlying_entry=Decimal("1000"),
            underlying_exit=Decimal("990"),
            cert_spec=bear_5x,
        )
        assert ret == Decimal("0.05")

    def test_bull_cert_loses_when_underlying_down(self, bull_5x: CertSpec) -> None:
        """1% down in underlying = 5% down in 5x bull cert."""
        _, ret = simulate_intraday_trade(
            underlying_entry=Decimal("1000"),
            underlying_exit=Decimal("990"),
            cert_spec=bull_5x,
        )
        assert ret == Decimal("-0.05")

    def test_exit_price_consistent_with_return(self, bull_5x: CertSpec) -> None:
        """Exit price should match return applied to initial price."""
        exit_price, ret = simulate_intraday_trade(
            underlying_entry=Decimal("1000"),
            underlying_exit=Decimal("1020"),
            cert_spec=bull_5x,
            initial_cert_price=Decimal("100"),
        )
        assert exit_price == Decimal("100") * (Decimal("1") + ret)


class TestCertPath:
    def test_empty_input_returns_empty_series(self, bull_5x: CertSpec) -> None:
        empty = pd.Series([], dtype=float)
        result = simulate_cert_path(empty, bull_5x)
        assert len(result) == 0

    def test_flat_underlying_only_loses_to_fees(self, bull_5x: CertSpec) -> None:
        """If underlying is flat, cert only loses the daily fee."""
        flat = pd.Series([1000.0] * 10, name="^OMX")
        result = simulate_cert_path(flat, bull_5x)
        # After ~10 days of 0.08% daily fee, cert is at ~99.2
        assert result.iloc[-1] < 100
        assert result.iloc[-1] > 99

    def test_volatility_decay_at_high_leverage(self) -> None:
        """A series that ends where it started should LOSE value in high-leverage cert.

        This is the volatility decay effect — daily reset means oscillating
        underlying drags the cert down over time.
        """
        # Underlying goes 100 -> 110 -> 100 -> 110 -> 100 (ends at start)
        oscillating = pd.Series([100.0, 110.0, 100.0, 110.0, 100.0], name="^OMX")
        high_lev = CertSpec(name="X10", underlying="^OMX", leverage=Decimal("10"))
        result = simulate_cert_path(oscillating, high_lev)
        # Cert should be below initial despite underlying being flat
        assert result.iloc[-1] < 100

    def test_clipping_prevents_negative_prices(self) -> None:
        """A move large enough to wipe out the cert is clipped to near-zero."""
        # 30% down in underlying on a 5x bull = -150% theoretical = wipeout
        crash = pd.Series([100.0, 70.0], name="^OMX")
        high_lev = CertSpec(name="X5", underlying="^OMX", leverage=Decimal("5"))
        result = simulate_cert_path(crash, high_lev)
        # Should be clipped, not negative
        assert result.iloc[-1] > 0
        assert result.iloc[-1] < 1  # But very small
