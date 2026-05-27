"""Unit tests for the market-data DB layer (no database required)."""

from __future__ import annotations

import pytest

from packages.market_data.db import MIGRATIONS, get_database_url


class TestGetDatabaseUrl:
    def test_raises_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
            get_database_url()

    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
        assert get_database_url() == "postgresql://u:p@localhost:5432/db"


class TestMigrations:
    def test_versions_are_unique(self) -> None:
        versions = [m.version for m in MIGRATIONS]
        assert len(versions) == len(set(versions))

    def test_versions_are_strictly_increasing(self) -> None:
        versions = [m.version for m in MIGRATIONS]
        assert versions == sorted(versions)
        assert versions == list(range(1, len(versions) + 1))

    def test_creates_bars_table_and_hypertable(self) -> None:
        sql_blob = " ".join(m.sql for m in MIGRATIONS).lower()
        assert "create table if not exists bars" in sql_blob
        assert "create_hypertable" in sql_blob
        assert "primary key (ticker, interval, ts)" in sql_blob

    def test_enables_timescaledb_extension(self) -> None:
        sql_blob = " ".join(m.sql for m in MIGRATIONS).lower()
        assert "create extension if not exists timescaledb" in sql_blob
