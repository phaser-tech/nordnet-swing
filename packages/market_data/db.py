"""Database layer for market data: connection, schema, and migrations.

Phase 0 uses **synchronous** psycopg3 on purpose. Historical ingestion is an
offline batch job, not a live critical path, so CLAUDE.md's "no sync I/O in
critical paths" rule does not apply here. The Phase 1 live feed will get its
own async ingestion path.

Storage is raw SQL by design: the schema is essentially one table plus a
TimescaleDB hypertable, whose DDL is raw SQL regardless. No ORM.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg
import structlog
from psycopg.rows import TupleRow

log = structlog.get_logger(__name__)


def get_database_url() -> str:
    """Return ``DATABASE_URL`` from the environment, or fail loudly.

    Per CLAUDE.md rule #7, connection details come from the environment, never
    from code. A missing URL is a configuration error we surface immediately.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Source it from .env (see .env.example) "
            "before connecting to the database."
        )
    return url


@contextmanager
def connect() -> Iterator[psycopg.Connection[TupleRow]]:
    """Open a synchronous connection from ``DATABASE_URL``, closing on exit.

    Transactions are the caller's responsibility: call ``conn.commit()`` after
    writes. The connection is closed when the context exits.
    """
    conn = psycopg.connect(get_database_url())
    try:
        yield conn
    finally:
        conn.close()


@dataclass(frozen=True)
class Migration:
    """A single forward-only schema migration.

    Migrations are applied in ascending ``version`` order and recorded in
    ``schema_migrations`` so the runner is idempotent. Each statement is also
    written to be safe to re-run on its own (``IF NOT EXISTS`` / ``if_not_exists``).
    """

    version: int
    name: str
    sql: str


# Forward-only migrations. Append new ones with the next version number;
# never edit or renumber an applied migration.
MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="enable_timescaledb",
        sql="CREATE EXTENSION IF NOT EXISTS timescaledb;",
    ),
    Migration(
        version=2,
        name="create_bars",
        # PK includes ts because TimescaleDB requires the partitioning column
        # to be part of any unique constraint on a hypertable.
        sql="""
        CREATE TABLE IF NOT EXISTS bars (
            ticker   text        NOT NULL,
            interval text        NOT NULL,
            ts       timestamptz NOT NULL,
            open     numeric     NOT NULL,
            high     numeric     NOT NULL,
            low      numeric     NOT NULL,
            close    numeric     NOT NULL,
            volume   bigint      NOT NULL,
            PRIMARY KEY (ticker, interval, ts)
        );
        """,
    ),
    Migration(
        version=3,
        name="bars_hypertable",
        sql="SELECT create_hypertable('bars', 'ts', if_not_exists => TRUE);",
    ),
)


def run_migrations() -> list[int]:
    """Apply any unapplied migrations in order. Idempotent.

    Returns the list of versions applied during this call (empty if the schema
    was already up to date).
    """
    applied_now: list[int] = []
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version    integer     PRIMARY KEY,
                    name       text        NOT NULL,
                    applied_at timestamptz NOT NULL DEFAULT now()
                );
                """
            )
            cur.execute("SELECT version FROM schema_migrations;")
            already_applied = {row[0] for row in cur.fetchall()}

            for migration in sorted(MIGRATIONS, key=lambda m: m.version):
                if migration.version in already_applied:
                    log.debug(
                        "migration.skip", version=migration.version, name=migration.name
                    )
                    continue
                log.info(
                    "migration.apply", version=migration.version, name=migration.name
                )
                cur.execute(migration.sql)
                cur.execute(
                    "INSERT INTO schema_migrations (version, name) VALUES (%s, %s);",
                    (migration.version, migration.name),
                )
                applied_now.append(migration.version)
        conn.commit()

    if applied_now:
        log.info("migrations.done", applied=applied_now)
    else:
        log.debug("migrations.up_to_date")
    return applied_now
