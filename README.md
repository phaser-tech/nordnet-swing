# Nordnet Swing

Selective swing-trading system for Nordnet Markets index certificates (Bull/Bear, Mini Futures, Unlimited Turbos) on OMX and Nasdaq 100.

**Status**: Early development — Phase 0 (backtest framework).

## What this is

A system that takes **1–5 trades per day** on high-conviction, multi-confluence setups, using leveraged certificates to capture multi-percent moves. Not high-frequency. Not day-scalping. Selective swing-intraday.

**Default state is to NOT trade.** Edge comes from being picky, not from being fast.

## What this is not

- Not a low-latency system. Nordnet's API has 100ms–1s response times.
- Not a market-making system.
- Not a system that trades individual stocks (only broad-index certificates).
- Not investment advice.

## Quick start

Prerequisites: Python 3.12, `uv`, Docker Desktop running.

```bash
# Install Python 3.12 for this project (uv manages this, doesn't touch system Python)
uv python install 3.12

# Install dependencies
uv sync

# Copy env template and edit if needed
cp .env.example .env

# Start PostgreSQL + TimescaleDB
docker compose up -d

# Verify
uv run pytest

# Run linter and type-check
uv run ruff check .
uv run mypy packages/
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for design principles and module boundaries.

## Working with Claude Code

This repo is structured for autonomous development with Claude Code. See [CLAUDE.md](CLAUDE.md) for project-specific instructions, conventions, and pitfalls.

## License

Private. All rights reserved.
