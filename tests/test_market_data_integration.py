"""Integration tests against the local Docker TimescaleDB.

Skipped automatically if the database is unreachable (e.g. CI without Docker),
so the unit suite stays green standalone. Run the stack with
``docker compose up -d`` to exercise these.

Uses a synthetic ticker so it never touches real ingested data, and cleans up
after itself.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal

import pandas as pd
import pytest
from dotenv import load_dotenv

from packages.market_data.db import connect, run_migrations
from packages.market_data.historical import DAILY_INTERVAL, get_bars, store_bars

load_dotenv()

TEST_TICKER = "ZZTEST.SYNTH"


def _db_available() -> bool:
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1;")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(),
    reason="local TimescaleDB not reachable (run `docker compose up -d`)",
)


def _canonical_frame(closes: list[float]) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        pd.to_datetime([f"2020-01-{2 + i:02d}" for i in range(len(closes))]),
        name="ts",
    ).tz_localize("UTC")
    return pd.DataFrame(
        {
            "Open": [c - 0.5 for c in closes],
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": closes,
            "Volume": [1000 + i for i in range(len(closes))],
        },
        index=index,
    )


@pytest.fixture(autouse=True)
def _clean_test_rows() -> Iterator[None]:
    run_migrations()
    _delete_test_rows()
    yield
    _delete_test_rows()


def _delete_test_rows() -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM bars WHERE ticker = %s;", (TEST_TICKER,))
        conn.commit()


def test_run_migrations_is_idempotent() -> None:
    # Already applied by the autouse fixture; a second run applies nothing.
    assert run_migrations() == []


def test_store_then_get_roundtrip() -> None:
    frame = _canonical_frame([100.0, 101.0, 102.0])
    written = store_bars(frame, TEST_TICKER, DAILY_INTERVAL)
    assert written == 3

    out = get_bars(TEST_TICKER, "2019-01-01", "2021-01-01", DAILY_INTERVAL)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(out) == 3
    assert out["Close"].tolist() == [100.0, 101.0, 102.0]
    assert str(out.index.tz) == "UTC"


def test_upsert_is_idempotent_no_duplicates() -> None:
    frame = _canonical_frame([100.0, 101.0])
    store_bars(frame, TEST_TICKER, DAILY_INTERVAL)
    store_bars(frame, TEST_TICKER, DAILY_INTERVAL)  # re-run

    out = get_bars(TEST_TICKER, "2019-01-01", "2021-01-01", DAILY_INTERVAL)
    assert len(out) == 2  # no duplication on PK conflict


def test_upsert_updates_revised_bars() -> None:
    store_bars(_canonical_frame([100.0, 101.0]), TEST_TICKER, DAILY_INTERVAL)
    # Same timestamps, revised closes — upsert should overwrite.
    store_bars(_canonical_frame([200.0, 201.0]), TEST_TICKER, DAILY_INTERVAL)

    out = get_bars(TEST_TICKER, "2019-01-01", "2021-01-01", DAILY_INTERVAL)
    assert out["Close"].tolist() == [200.0, 201.0]


def test_numeric_precision_preserved_in_db() -> None:
    # A price with cents must round-trip exactly via numeric (not binary float).
    frame = _canonical_frame([100.0])
    frame.iloc[0, frame.columns.get_loc("Close")] = 1234.57
    store_bars(frame, TEST_TICKER, DAILY_INTERVAL)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT close FROM bars WHERE ticker = %s ORDER BY ts;",
            (TEST_TICKER,),
        )
        (db_close,) = cur.fetchone()  # type: ignore[misc]
    assert db_close == Decimal("1234.57")
