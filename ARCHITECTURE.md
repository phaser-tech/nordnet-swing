# Architecture

## Design principles

### 1. Event-driven, modular, single-VPS-deployable
The system runs as multiple processes communicating via an event bus. Each process is independently startable, restartable, and testable. We don't optimize for horizontal scaling — single user, single capital pool. We optimize for **reliability**, **observability**, and **safety**.

### 2. Source of truth lives in Nordnet
Local state is always a projection. Every order, position, and fill exists authoritatively in Nordnet's systems. We reconcile against that ground truth on every restart and continuously in production.

### 3. Strict layer separation
```
services/        (deployable processes — main.py entry points)
   ↓ imports
packages/        (libraries — domain logic, infrastructure adapters)
   ↓ imports
packages/core/   (shared types and utilities — no business logic)
```
`packages/` does NOT import from `services/`. Packages don't import from each other except via `core/`.

### 4. Domain types are frozen
All types in `packages/core/domain/` are immutable pydantic models. Mutation happens through state machines that produce new instances. This makes reasoning about concurrency and debugging dramatically easier.

### 5. Idempotency at every boundary
Every order has a `client_order_id` we generate before submission. Every state transition is logged with that id. Retries are safe.

## Module responsibilities

### `packages/core/`
Shared types, events, time utilities. No business logic. Imported everywhere.

### `packages/nordnet_client/` (Phase 1)
Pure infrastructure. Authenticates against Nordnet, handles REST + WebSocket feeds, normalizes wire format to domain types. Knows nothing about strategies, decisions, or risk.

### `packages/market_data/` (Phase 1)
Ingests data from nordnet_client into TimescaleDB. Provides query and replay interfaces. Same interface for live and backtest — strategies don't know which mode they're running in.

### `packages/strategies/`
Pluggable strategies. Each implements `Strategy.generate_signals(market_data) -> Iterable[Signal]`. Strategies are stateless; state lives elsewhere.

### `packages/decision/`
Aggregates signals across strategies. Applies confluence scoring. Calls cost engine. Decides whether a Signal becomes an OrderIntent.

### `packages/execution/` (Phase 3)
Owns communication with Nordnet for orders. Manages order lifecycle state machine. Implements idempotent submission, cancel/replace, timeout.

### `packages/risk/` (Phase 3)
Hard limits enforcement, kill switch, knockout monitoring, anomaly detection. Can publish `Halt` events that every other component respects.

### `packages/backtest/`
Phase 0 priority. Simulator + cost model + strategy runner + performance analyzer. Uses the same strategy code as live — only data source and execution differ.

## Data flow (live)

```
Nordnet feeds
   ↓
nordnet_client (normalizes)
   ↓
market_data (stores + publishes events)
   ↓
strategies (generate signals)
   ↓
decision (filters via confluence + cost)
   ↓
execution (submits to Nordnet)
   ↓
nordnet_client (confirms via private feed)
   ↓
market_data (updates positions)
   ↓
reconciler (verifies match)

risk supervisor watches everything; can publish Halt at any point.
```

## Data flow (backtest)

```
Historical data files / yfinance
   ↓
market_data.replay (publishes events as if live)
   ↓
strategies (same code as live)
   ↓
decision (same code as live)
   ↓
execution simulator (instead of real Nordnet)
   ↓
performance analyzer
```

The strategy code is identical between live and backtest. This is intentional and load-bearing — backtest results that don't transfer are useless.

## State management

- **Source of truth**: Nordnet (orders, positions, fills)
- **Local projection**: PostgreSQL (orders table, positions table, trades table)
- **Reconciliation**: every 30 seconds + on every restart
- **On drift detected**: publish `Halt` event, require manual intervention

## Persistence

- PostgreSQL with TimescaleDB extension on a single VPS
- TimescaleDB for tick/bar data (time-series)
- Regular PostgreSQL tables for orders, positions, trades, audit log
- No separate analytics DB — same data, just different views
- Daily backups via Hetzner snapshot

## Observability

- Structured JSON logs (structlog)
- Prometheus metrics on every service
- Grafana dashboards: feed status, signal generation rate, cost-blocked trades, position state, P&L
- Alerts on: feed disconnects, reconciliation drift, kill-switch activation, P&L drawdown limits

## Security boundaries

- Nordnet credentials live in environment variables sourced from encrypted `.env` on VPS
- No credentials in code, repo, or logs
- Database not exposed to public internet (only localhost access)
- Dashboard behind TLS + simple auth (single user, can be basic)
- Audit log is append-only

## What we deliberately don't have

- Kubernetes — overkill for a single-user system
- Microservices in the classic sense — we have processes, not networked services
- Multiple databases — one PostgreSQL handles everything
- Service mesh, API gateway, distributed tracing — none needed at this scale
- Auto-scaling — fixed capacity, no need

## Open architectural questions

These are deferred until relevant:

- Should we add a second VPS for hot failover? (Probably no — manual restart is fine for personal use)
- Multiple brokers in the future? (If yes, abstract behind a `BrokerClient` interface from the start — but only if there's concrete demand)
- Stream processing framework? (No — pandas + asyncio handles our volume)
