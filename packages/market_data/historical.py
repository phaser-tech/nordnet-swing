"""Historical (batch) market-data ingestion from yfinance into TimescaleDB.

Pipeline: ``fetch_bars`` (yfinance) -> ``store_bars`` (idempotent upsert) ->
``get_bars`` (read back). ``sync`` runs fetch + store for a set of tickers.

Column-case convention: the DataFrame API uses capitalized OHLCV columns
(``Open/High/Low/Close/Volume``), matching both yfinance's native output and
the documented ``Strategy.generate_signals`` contract. The DB columns are
lowercase; ``store_bars``/``get_bars`` map between the two.

Precision convention (CLAUDE.md): the ``Bar`` boundary model and the DB
columns use ``Decimal``/``numeric`` to preserve precision. The DataFrame API
exposes ``float`` because that is what pandas math and the strategy layer use
("float only for indicators/math").

Intervals: ``1d`` and ``1h`` are supported. yfinance returns up to ~730 days of
1h history in a single window — enough for the first round of Tier-2 intraday
OOS tests on ``^OMX`` (issue #16). Sub-hour granularity (``1m``, ``5m``) and
deeper 1h history both need windowed/incremental fetching and stay deferred.

Bars come back at top-of-hour with whatever the source venue published; for
Stockholm-listed indices the last bar of the trading day is often a partial
covering the 17:00-17:30 CET close — we keep it as observed and let strategies
filter if they need to.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pandas as pd
import structlog
import yfinance as yf
from pydantic import BaseModel, ConfigDict

from packages.market_data.db import connect

log = structlog.get_logger(__name__)

# Cross-asset universe we hunt edges across (CLAUDE.md "edge sources"):
# OMX + Nasdaq + S&P (equity), VIX (vol), DXY + 10y yield (macro), gold (risk).
DEFAULT_INSTRUMENTS: tuple[str, ...] = (
    "^OMX",  # OMX Stockholm 30 (Yahoo symbol; ^OMXS30 returns no data)
    "^NDX",  # Nasdaq 100
    "^GSPC",  # S&P 500
    "^VIX",  # CBOE Volatility Index
    "DX-Y.NYB",  # US Dollar Index
    "^TNX",  # US 10-year Treasury yield
    "GC=F",  # Gold futures
)

DAILY_INTERVAL = "1d"
HOURLY_INTERVAL = "1h"
# yfinance gives ~10y for daily and ~730d for 1h in a single window. Sub-hour
# and beyond-730d intraday are deferred (windowed fetch — see module docstring).
SUPPORTED_INTERVALS: frozenset[str] = frozenset({DAILY_INTERVAL, HOURLY_INTERVAL})

# Canonical OHLCV columns for the DataFrame API, in storage order.
_OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


class Bar(BaseModel):
    """A single OHLCV bar. Boundary type crossing the DB process boundary.

    Immutable, and uses ``Decimal`` for prices per the money/price rule.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    interval: str
    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


def _validate_interval(interval: str) -> None:
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(
            f"Unsupported interval {interval!r}. Supported: "
            f"{sorted(SUPPORTED_INTERVALS)}. Intraday is deferred to a later issue."
        )


def _to_decimal(value: object) -> Decimal:
    """Convert a numpy/pandas scalar to Decimal without float artifacts."""
    return Decimal(str(value))


def _empty_frame() -> pd.DataFrame:
    """An empty OHLCV frame with the canonical columns and a ts index."""
    frame = pd.DataFrame(columns=list(_OHLCV_COLUMNS))
    frame.index = pd.DatetimeIndex([], tz="UTC", name="ts")
    return frame


def _normalize_yf_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a raw yfinance frame to the canonical OHLCV shape.

    Handles yfinance returning either single-level columns or a ``(field,
    ticker)`` MultiIndex for single-ticker downloads. Produces a frame indexed
    by a UTC-aware ``ts`` index with exactly the ``_OHLCV_COLUMNS``, NaN rows
    dropped.
    """
    if raw.empty:
        return _empty_frame()

    frame = raw.copy()

    # Collapse a MultiIndex (field, ticker) down to single-level field columns.
    if isinstance(frame.columns, pd.MultiIndex):
        for level in range(frame.columns.nlevels):
            if ticker in frame.columns.get_level_values(level):
                frame = frame.xs(ticker, axis=1, level=level)
                break
        else:
            frame.columns = frame.columns.get_level_values(0)

    missing = [col for col in _OHLCV_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(
            f"yfinance frame for {ticker!r} is missing columns {missing}; "
            f"got {list(frame.columns)}"
        )

    frame = frame.loc[:, list(_OHLCV_COLUMNS)].copy()

    # Normalize the index to a UTC-aware DatetimeIndex named "ts".
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    frame.index = index.rename("ts")

    frame = frame.dropna(subset=list(_OHLCV_COLUMNS))
    return frame


def fetch_bars(
    ticker: str,
    start: date | datetime,
    end: date | datetime,
    interval: str = DAILY_INTERVAL,
) -> pd.DataFrame:
    """Fetch bars for ``ticker`` over ``[start, end)`` from yfinance.

    Returns a DataFrame indexed by a UTC-aware ``ts`` index with capitalized
    OHLCV columns. ``auto_adjust=False`` keeps raw OHLC (we store unadjusted
    prices; for indices/futures here, adjustment is largely a no-op anyway).
    """
    _validate_interval(interval)
    raw = yf.download(
        tickers=ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        log.warning("fetch.empty", ticker=ticker, start=str(start), end=str(end))
        return _empty_frame()
    return _normalize_yf_frame(raw, ticker)


def _frame_to_bars(frame: pd.DataFrame, ticker: str, interval: str) -> list[Bar]:
    """Convert a normalized OHLCV frame into validated ``Bar`` models."""
    bars: list[Bar] = []
    for ts, row in frame.iterrows():
        bars.append(
            Bar(
                ticker=ticker,
                interval=interval,
                ts=ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts,
                open=_to_decimal(row["Open"]),
                high=_to_decimal(row["High"]),
                low=_to_decimal(row["Low"]),
                close=_to_decimal(row["Close"]),
                volume=int(row["Volume"]),
            )
        )
    return bars


def store_bars(bars: pd.DataFrame, ticker: str, interval: str) -> int:
    """Upsert a normalized OHLCV frame into the ``bars`` table. Idempotent.

    On primary-key conflict ``(ticker, interval, ts)`` the row is updated, so
    re-running ``sync`` corrects any revised bars instead of erroring or
    duplicating. Returns the number of rows written.
    """
    _validate_interval(interval)
    if bars.empty:
        log.debug("store.skip_empty", ticker=ticker, interval=interval)
        return 0

    rows = [
        (
            bar.ticker,
            bar.interval,
            bar.ts,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
        )
        for bar in _frame_to_bars(bars, ticker, interval)
    ]

    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO bars
                    (ticker, interval, ts, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, interval, ts) DO UPDATE SET
                    open   = EXCLUDED.open,
                    high   = EXCLUDED.high,
                    low    = EXCLUDED.low,
                    close  = EXCLUDED.close,
                    volume = EXCLUDED.volume;
                """,
                rows,
            )
        conn.commit()

    log.info("store.done", ticker=ticker, interval=interval, rows=len(rows))
    return len(rows)


def get_bars(
    ticker: str,
    start: date | datetime,
    end: date | datetime,
    interval: str = DAILY_INTERVAL,
) -> pd.DataFrame:
    """Read bars for ``ticker`` over ``[start, end)`` back from the DB.

    Returns the same canonical shape as ``fetch_bars`` (UTC ``ts`` index,
    capitalized OHLCV columns). OHLC come back as ``float`` for the DataFrame
    API; precision is preserved in the DB's ``numeric`` columns.
    """
    _validate_interval(interval)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT ts, open, high, low, close, volume
            FROM bars
            WHERE ticker = %s AND interval = %s AND ts >= %s AND ts < %s
            ORDER BY ts;
            """,
            (ticker, interval, start, end),
        )
        records = cur.fetchall()

    if not records:
        return _empty_frame()

    frame = pd.DataFrame(
        records, columns=["ts", "Open", "High", "Low", "Close", "Volume"]
    ).set_index("ts")
    for col in ("Open", "High", "Low", "Close"):
        frame[col] = frame[col].astype(float)
    frame["Volume"] = frame["Volume"].astype("int64")
    frame.index = pd.DatetimeIndex(frame.index).rename("ts")
    return frame


def sync(
    tickers: tuple[str, ...] | list[str],
    start_date: date | datetime,
    interval: str = DAILY_INTERVAL,
    end_date: date | datetime | None = None,
) -> dict[str, int]:
    """Fetch + store bars for each ticker. Returns ``{ticker: rows_written}``.

    ``end_date`` defaults to today (UTC). The fetch window is half-open
    ``[start_date, end_date)``.
    """
    _validate_interval(interval)
    end = end_date if end_date is not None else datetime.now(UTC).date()

    results: dict[str, int] = {}
    for ticker in tickers:
        log.info(
            "sync.fetch",
            ticker=ticker,
            start=str(start_date),
            end=str(end),
            interval=interval,
        )
        frame = fetch_bars(ticker, start_date, end, interval)
        results[ticker] = store_bars(frame, ticker, interval)

    log.info("sync.done", total_rows=sum(results.values()), tickers=len(results))
    return results
