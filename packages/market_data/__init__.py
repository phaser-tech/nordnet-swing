"""Market data: ingestion, storage, and query interfaces.

Phase 0 provides historical (batch) ingestion from yfinance into a
TimescaleDB ``bars`` hypertable. Live streaming ingestion arrives in Phase 1
and will get its own async path; see ``ARCHITECTURE.md``.
"""
