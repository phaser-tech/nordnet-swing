# Claude Code Instructions for Nordnet Swing

Read this file first in every session. It contains project-specific context that supersedes general defaults.

## What we're building

A selective swing-trading system that takes **1–5 trades per day** on high-conviction setups in Nordnet Markets index certificates (Bull/Bear, Mini Futures, Unlimited Turbos) on OMX and Nasdaq 100. We use leverage (5–15x) to capture multi-percent moves.

**Not** high-frequency trading. **Not** day-scalping. The system's job is to *not trade* most of the time, and to act decisively when multiple confluence factors align.

## Hard architecture rules

These are non-negotiable. Surface a question before violating one of these:

1. **Nordnet is always the source of truth.** Local state is a projection. Reconciliation must pass before any new trade.
2. **Never market orders. Always limit orders.** No exceptions.
3. **Cost engine must pass before signal escalates to order.** A signal without a passing cost check is not a trade candidate.
4. **Every order needs a `client_order_id` (UUID) generated locally before submission.** This enables idempotent retry.
5. **Default state is no-trade.** Bias every design decision toward selectivity over coverage.
6. **`packages/` may not import from `services/`.** `services/` may import from `packages/`. Packages may only import `core/` and their own modules.
7. **No secrets in code, logs, or test fixtures.** Use `os.environ` with sensible failures.
8. **Strict typing.** `mypy --strict` must pass. No `Any` without an inline comment explaining why.

## Code conventions

- Python 3.12, `uv` for package management
- `pydantic` for all data crossing process boundaries (API, DB, IPC)
- `asyncio` / `anyio` for I/O. No sync calls in critical paths.
- `structlog` for logging, JSON in production, console in dev
- Type hints on every function signature
- Domain types live in `packages/core/domain/`, are frozen pydantic models
- `Decimal` for money/prices in domain layer. `float` only for indicators/math.

## Testing requirements

- `pytest` with `hypothesis` for property-based tests on domain logic
- Aim for 80%+ coverage on `packages/decision/`, `packages/execution/`, `packages/risk/` (safety-critical)
- Every new strategy MUST come with a backtest report in the PR description: EV after costs, hit rate, max drawdown, equity curve vs OMX buy-and-hold
- Integration tests against Nordnet's sandbox before any production change

## Workflow expectations

1. **Create a GitHub issue first**, even for small changes. State the goal and acceptance criteria.
2. **One PR per logical change.** Small diffs review faster.
3. **Run `uv run ruff check . && uv run mypy packages/ && uv run pytest` before pushing.** CI will fail on these otherwise.
4. **Commit messages**: conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`)
5. **Never auto-merge.** Trading systems deserve manual gatekeeping.

## Domain knowledge (non-obvious)

### Nordnet Markets cost structure
- **Courtage = 0** for Nordnet Markets products via Next API when order > 1000 SEK
- Real cost = issuer spread (0.3–0.8% round-trip normal, up to 2%+ during news)
- Spread widens 3–5x during major news releases — pause trading in those windows
- See `packages/backtest/cost_model.py` for the model we use

### Leveraged certificates have daily reset
- Bull/Bear certificates apply leverage to *daily* returns, then reset
- Holding longer than 1 day introduces compounding drag (volatility decay)
- We never hold overnight (rules out overnight financing AND volatility drag)
- Cert price simulator in `packages/backtest/simulator.py` handles this correctly

### Nordnet API quirks
- nExt API has 100ms–1s latency. Don't design for sub-second strategies.
- Socket feeds can drop without warning. Every consumer needs reconnect + exponential backoff.
- Subscription cap of ~100 instruments per session.
- Production access requires personal certification at Nordnet — not a self-service flow.
- Some endpoints return XML, not JSON. Use the parser in `packages/nordnet_client/` (once built).

### Edge sources we're hunting
1. **Macro-calendar plays** — pre-positioning when consensus is skewed; fade initial reactions when overreacting
2. **Cross-asset confirmation** — bonds/USD/gold/VIX leading equity moves
3. **Technical confluence at daily key levels** — high-volume tests of multi-day swings + macro bias

### Edge sources we're NOT hunting
- Sub-second breakouts (latency-bound)
- Pure mean reversion without macro filter
- Overnight holds in leveraged certificates
- Individual stock certificates (idiosyncratic risk)

## Common pitfalls to avoid

- **Don't add a new strategy without backtest validation first.** Edge unproven = code wasted.
- **Don't catch broad exceptions in execution code.** Log + re-raise. Failing loudly beats failing silently when money moves.
- **Don't trust local position state during incidents.** When in doubt, reconcile against Nordnet.
- **Don't use Python `float` for prices in domain layer.** Use `Decimal`. Float math will eventually bite a P&L number.
- **Don't optimize for backtest performance.** Optimize for *out-of-sample* performance. Walk-forward, always.
- **Don't add CI/CD auto-deploy.** Production deploys are manually triggered.

## How to ask good questions

When you're unsure about a design choice, surface it with:
- What you're trying to do
- Two or three options you considered
- The trade-offs you see
- Your recommendation

Don't guess on architecture decisions. Don't silently choose. We'd rather pause and align.

## Definitely not allowed

- Writing trading credentials to disk in plaintext
- Hard-coding instrument symbols outside config
- Live trading without a kill-switch in scope
- Disabling tests to make CI green
- Committing data files larger than 1MB

## Phase status

We are in **Phase 0**: backtest framework + strategy validation.

Phases ahead:
1. Phase 1 — Nordnet client + market data ingestion
2. Phase 2 — Decision engine + cost-aware execution simulator
3. Phase 3 — Live execution gateway + risk supervisor + reconciler
4. Phase 4 — Production deployment + paper trading
5. Phase 5 — Live with minimum capital

Don't skip ahead. Don't build Phase 2 components in Phase 0.
