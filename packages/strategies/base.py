"""Base class for trading strategies.

A Strategy is a pure function from market data to signals. It MUST:
- Be stateless (state lives in market_data or external services)
- Produce immutable Signal objects
- Document the confluence_factors that triggered each signal
- Use the same code in backtest and live

A Strategy MUST NOT:
- Reach out to the network
- Decide whether a trade happens (that's the decision engine)
- Care about position sizing (that's the decision engine)
- Care about cost (that's the cost engine)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import pandas as pd

from packages.core.domain.signal import Signal


class Strategy(ABC):
    """Abstract base for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier. Used in logs, signal attribution, configs."""
        ...

    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame) -> Iterable[Signal]:
        """Generate signals from market data.

        Args:
            market_data: DataFrame indexed by datetime with at minimum the
                         columns Open, High, Low, Close, Volume.
                         Additional columns (cross-asset, regime) may be
                         present depending on context.

        Yields:
            Signal objects. May be empty if no signals on this bar.

        Implementation note: prefer generator functions for memory efficiency
        in backtests over multi-year data.
        """
        ...
