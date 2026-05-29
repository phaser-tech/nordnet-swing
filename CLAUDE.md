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
9. **No new strategy without OOS validation.** The `oos.py` harness exists for a reason. Use it. Frozen params on train, blind apply to test, both gross and net reported.

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
- Every new strategy MUST come with an OOS report in the PR description: train/test split, frozen config, gross AND net metrics, EV after costs, hit rate, max drawdown, equity curve vs OMX buy-and-hold
- Integration tests against Nordnet's sandbox before any production change

## Workflow expectations

1. **Create a GitHub issue first**, even for small changes. State the goal and acceptance criteria.
2. **One PR per logical change.** Small diffs review faster.
3. **Run `uv run ruff check . && uv run mypy packages/ && uv run pytest` before pushing.** CI will fail on these otherwise.
4. **Commit messages**: conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`)
5. **Never auto-merge.** Trading systems deserve manual gatekeeping.
6. **Pre-register decision rules** for OOS tests before seeing test data. If pre-registered, honor them — don't rationalize positives that weren't there.

## Domain knowledge (non-obvious)

### Nordnet Markets cost structure
- **Courtage = 0** for Nordnet Markets products via Next API when order > 1000 SEK
- Real cost = issuer spread (0.3–0.8% round-trip normal, up to 2%+ during news)
- Spread widens 3–5x during major news releases — pause trading in those windows
- See `packages/backtest/cost_model.py` for the model we use
- **Important caveat**: our model assumes spread is fixed % of cert position regardless of leverage. In reality, higher-leverage certs (10x+) carry wider spreads. Do not use the cost model to justify leverage choices > 5x without Phase 1 real-spread validation.

### Leveraged certificates have daily reset
- Bull/Bear certificates apply leverage to *daily* returns, then reset
- Holding longer than 1 day introduces compounding drag (volatility decay)
- We never hold overnight as default (see Strategic tensions below for the caveat)
- Cert price simulator in `packages/backtest/simulator.py` handles this correctly

### Nordnet API quirks
- nExt API has 100ms–1s latency. Don't design for sub-second strategies.
- Socket feeds can drop without warning. Every consumer needs reconnect + exponential backoff.
- Subscription cap of ~100 instruments per session.
- Production access requires personal certification at Nordnet — not a self-service flow.
- Some endpoints return XML, not JSON. Use the parser in `packages/nordnet_client/` (once built).

### OMX ticker on Yahoo Finance
- Use `^OMX` for OMX Stockholm 30 price index (NOT `^OMXS30`, which returns no data on yfinance).
- `^OMX` is the OMXS30 index — confusingly named but verified against Nasdaq official data and via return-correlation analysis against the XACT-OMXS30.ST ETF.
- The broader All-Share index is the separate `^OMXSPI` ticker. Don't confuse them.

## Edge sources we're hunting

The original menu (3 sources) was too narrow — two of three failed OOS using daily-bar architecture. This expanded list is tiered by current testability.

### Tier 1 — Testable now (Phase 0, daily bars only)

1. **Macro-event window selection.** Identify days with major scheduled macro events (CPI, FOMC, ECB, Riksbanksbesked, NFP, PMI). Trade open→close *only* on event days, predicting direction from consensus skew, pre-event positioning, or systematic post-event drift. Naturally selective (~10–20 event days/year per major series).

2. **Volatility regime as meta-filter.** Use VIX level + structure to classify days as trending vs chopping. Don't trade as a standalone signal — use it to *gate* other signals. Many strategies that fail across all regimes succeed within a specific regime.

3. **Mean reversion after extreme daily moves.** When OMX has had a 2+ sigma daily move (calculated rolling), the following day has a documented tendency to revert. Selective by construction (~5–15 occurrences/year). Test as both standalone signal and meta-trigger.

4. **Calendar effects.** Turn-of-month, day-of-week, post-holiday — individually small but well-documented. May combine with other signals to lift edge above costs.

5. **Bond/equity correlation regime breaks.** When normal SPX/US10Y correlation breaks down (both fall together, or both rise sharply), it often signals regime shift. Trackable with rolling correlation from data we already have.

### Tier 2 — Need intraday data (Phase 1 territory)

6. **Event-window intraday trading.** Trade only in the 30–60 min window after major macro releases, capturing initial overreaction → fade. Documented in academic literature. Requires intraday timing.

7. **Intraday technical patterns at key levels.** The original "technical confluence" idea, but with real intraday execution: test of level + observed reaction + entry in real time.

8. **First/last hour dynamics.** Opening range breaks, lunch chop, closing drift — each has documented statistical patterns. Selective and time-bound.

### Tier 3 — Need additional data sources

9. **Sentiment/positioning extremes.** COT data (CFTC), AAII survey, options gamma exposure (SpotGamma-like). Extreme positioning often precedes regime shifts.

10. **Cross-asset confluence (intraday timing).** Same signal we tested in Phase 0 (failed there), but with intraday execution that might capture moves before they're fully gap-priced.

## Empirical findings (institutional memory)

What we've actually tested and learned. Don't re-test dead configurations without changing the variable that caused failure.

### Strategies tested

| Strategy | Configuration | Result | PR |
|---|---|---|---|
| SMA crossover | 5x, daily, open→close | No edge (plumbing test, expected) | #4 |
| Cross-asset confluence | 5x, daily, open→close, k=0.5 | Thin in-sample (t=1.87), failed OOS (t=1.33, net Sharpe 0.02) | #8, #10 |
| Volume-confirmed breakout | 5x, daily, 20-day Donchian, 1.5x vol | Negative gross in train AND test (t=-0.18, t=-0.22). Predicts OPPOSITE direction. | #11 |

### Diagnostic pattern (important)

Two independent strategies (cross-asset macro signal + technical breakout signal) failed via the *same mechanism*: information is priced into the open gap before our open→close window. The breakout strategy actually had negative gross return — meaning the signal predicts the opposite of what it should — because by the time we trade, the move has already happened in the gap.

This is a structural finding, not a strategy-specific failure:
- Daily-bar signals using prior-day info systematically miss the move
- Open→close execution on day T cannot capture edge that lives in close(T-1)→open(T)
- This applies to ANY strategy of the form "use yesterday's info to predict today's open→close direction" on a market like OMX that opens after the relevant US session

### Implications for new strategies

- **Be skeptical** of daily-bar signals using prior-day info to predict same-day open→close direction. The gap eats them.
- **Prefer** event-driven or regime-conditional designs over continuous-signal designs. Selectivity matters more than coverage.
- **Or** pursue intraday execution (Phase 1) where signal timing can match information arrival.

## Strategic tensions (honest design conflicts)

Document where our design rules conflict with empirically observed reality. Surface these when designing new strategies.

### No-overnight rule vs gap-located edge

The no-overnight rule was established to avoid volatility decay on leveraged certificates over multi-day holds. But two independent Phase 0 strategies failed because edge lives in the overnight gap that this rule forbids us to capture.

**Open question**: should we reconsider for narrowly-scoped single-night gap-capture trades? A one-night hold suffers minimal vol decay (decay compounds over many days of oscillation; one directional night is minor). Financing for one night is negligible. This is a legitimate design conversation that the diagnostic pattern now justifies opening.

When you face this tension, surface it rather than silently extending or violating the rule.

### Daily bars only vs intraday-native edge sources

Phase 0 explored what daily resolution can offer. Several edge sources (event-window, intraday technicals, first/last hour) naturally require intraday data. Phase 1 unlocks this but requires significant infrastructure investment (Nordnet client, intraday ingestion, possibly server provisioning).

Don't propose intraday strategies in Phase 0 — they can't be honestly tested with what we have. Surface them as Phase 1 candidates instead.

### 5x default vs leverage diluting fixed cost

The cost model suggests higher leverage improves edge/cost ratio (fixed % cost spread over smaller notional → more market exposure per cost unit). This math is correct *given the model* but the model assumes constant spread % across leverage levels, which is false in reality (higher-leverage certs have wider spreads).

Do not use the model to justify > 5x leverage without real-spread validation. The "free lunch" of leverage diluting cost is largely a model artifact.

## Edge sources we're NOT hunting

- Sub-second breakouts (latency-bound — we have 100ms–1s API latency)
- Pure mean reversion without macro/regime filter (loses in trending regimes)
- Default overnight holds in leveraged certs (the rule, though see Strategic tensions for narrow exceptions)
- Earnings or single-stock cert strategies (idiosyncratic risk, narrow focus)
- Correlated trades (long Nasdaq bull + long OMX bull = same trade)
- Same-day open→close prediction from prior-day macro/cross-asset signals (empirically dead — gap-arbitraged, see Findings)
- Continuous always-on directional signals (cumulative cost over many trades kills any thin edge — confirmed by cross-asset 1267-trade backtest)

## Common pitfalls to avoid

- **Don't add a new strategy without OOS validation.** Edge unproven = code wasted.
- **Don't tune k or other parameters on the test set.** That's exactly the overfitting the OOS harness exists to prevent.
- **Don't catch broad exceptions in execution code.** Log + re-raise. Failing loudly beats failing silently when money moves.
- **Don't trust local position state during incidents.** When in doubt, reconcile against Nordnet.
- **Don't use Python `float` for prices in domain layer.** Use `Decimal`. Float math will eventually bite a P&L number.
- **Don't optimize for backtest performance.** Optimize for *out-of-sample* performance. Walk-forward, always.
- **Don't add CI/CD auto-deploy.** Production deploys are manually triggered.
- **Don't justify leverage choices > 5x from the cost model alone.** The model under-prices high-leverage spreads. Real-spread validation in Phase 1 required.
- **Don't propose strategies that violate our design rules without surfacing the tension first.** Especially no-overnight and limit-only-orders.

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

We are at the **end of Phase 0**: framework complete, three strategies tested via OOS, daily-bar edge exhausted with current architecture.

### Completed
- Phase 0: backtest framework + cost model + OOS harness + 3 strategies tested

### Active decisions
Before moving to Phase 1, the user is evaluating three paths:
- **A**: Phase 1 enligt plan (Nordnet client + intraday data) — test intraday-precision strategies
- **B**: Reconsider no-overnight rule for gap-capture trades, given the diagnostic findings
- **C**: Explore more Tier 1 edge sources (event-window, vol regime, mean reversion, calendar effects, correlation regime) which are still testable in Phase 0

When the user picks a path, that becomes the active phase. Until then, no new strategy work proceeds.

### Phases ahead (if continuing)
1. Phase 1 — Nordnet client + intraday market data ingestion
2. Phase 2 — Decision engine + cost-aware execution simulator (with intraday)
3. Phase 3 — Live execution gateway + risk supervisor + reconciler
4. Phase 4 — Production deployment + paper trading
5. Phase 5 — Live with minimum capital

Don't skip ahead. Don't build Phase 2 components in Phase 0.
