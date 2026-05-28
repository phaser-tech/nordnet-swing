"""Wide-frame assembly for multi-instrument strategies (composition-root helper).

Builds the single wide DataFrame a cross-asset strategy consumes: the traded
instrument's OHLCV plus each cross-asset's Close as a prefixed column
(`SPX_Close`, ...), all aligned on the traded instrument's trading calendar.

Rule #6: this lives in `backtest` but imports no sibling package — bars arrive
through the injected `BarsLoader` (the CLI passes `market_data.get_bars`). So it
stays a pure, DB-free, unit-testable helper.

Calendar alignment: cross-asset closes are reindexed onto the traded
instrument's trading days and forward-filled. On a day the foreign market was
closed (e.g. a US holiday), the "most recent available close" carries forward —
which, combined with the strategy's one-day shift, models "the latest US close
known at Stockholm's open."
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from packages.backtest.runner import BarsLoader

# Composition config: cross-asset column prefix -> yfinance ticker.
CROSS_ASSET_TICKERS: dict[str, str] = {
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "TNX": "^TNX",
    "DXY": "DX-Y.NYB",
    "GOLD": "GC=F",
}


def assemble_cross_asset_frame(
    *,
    loader: BarsLoader,
    instrument: str,
    start: date | datetime,
    end: date | datetime,
    interval: str,
    cross_assets: dict[str, str] = CROSS_ASSET_TICKERS,
) -> pd.DataFrame:
    """Assemble the wide frame: instrument OHLCV + `{PREFIX}_Close` per cross-asset.

    Cross-asset closes are aligned to the instrument's index (reindex + ffill).
    Returns an empty frame if the instrument has no bars.
    """
    base = loader(instrument, start, end, interval)
    if base.empty:
        return base

    frame = base.copy()
    for prefix, ticker in cross_assets.items():
        bars = loader(ticker, start, end, interval)
        close = bars["Close"] if not bars.empty else pd.Series(dtype=float)
        frame[f"{prefix}_Close"] = close.reindex(frame.index).ffill()
    return frame
