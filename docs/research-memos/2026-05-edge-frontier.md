# Edge frontier memo — where might genuine retail edge live for Swedish index instruments?

**Author**: research collaborator (Anders + Claude)
**Date**: 2026-05-31
**Branch**: `research/edge-frontier-memo`
**Status**: DRAFT — discussion document, no work to follow without explicit go-ahead.

---

## 1. Executive summary

Based on five OOS failures (PRs #10/#14/#15/#20/#22), one signal-source comparison (the EU/US analysis), and the academic literature on retail-accessible anomalies, **three directions stand out in priority order**: **(A) switch from leveraged certificates to OMXS30 index futures** — our cost wall is mostly an instrument choice, not a market fact, and futures collapse it by ~50–150×; **(B) capture the well-documented overnight return premium directly** (Lou-Polk-Skouras 2019; Knuteson 2020), now feasible under the ratified narrow no-overnight exception; **(C) test 2–5 day momentum + a portfolio-of-weak-signals construction** — the cost wall scales weakly with holding period while edge typically scales linearly in the persistence window, and Grinold's IR = IC · √breadth says even thin signals compound when diversified. **What we should explicitly STOP doing**: any further open→close daily-bar OMX strategy on a single signal at 5x cert leverage. That envelope is empirically dead with five independent data points.

---

## 2. What the evidence tells us

### Confidently falsified (Phase 0/1, five tests)
- **Daily-bar open→close on `^OMX` at 5× cert leverage cannot extract net edge** from: cross-asset macro confluence (PR #10), volume-confirmed Donchian breakout (PR #15), mean-reversion after ≥2σ daily moves (PR #14), opening-range break on 1h bars (PR #20), and — most importantly — *a signal that demonstrably has gap-direction agreement train→test* (cross-asset gap-capture, PR #22). The last result is the diagnostic: the cross-asset signal *correctly* identifies overnight gap direction (+0.065% mean in signal direction, train and test agree exactly), but the gap edge is the same order of magnitude as the 0.63% round-trip cost at 5× (breakeven 0.126% underlying).
- **Signal source is not the bottleneck**: the analysis branch (`analysis/omx-signal-source-comparison`) shows US signals dominate EU signals for OMX open→close in every rolling 2-year window, and adding EU adds zero adj-R² after dof penalty. The dominant signal class never flipped. Best multivariate R² across all windows: ~3% typical, ~8% best regime. The signal sets we tested have approximately the predictive power they have; we are not missing a magic European factor.
- **Information arrives largely in the close(T-1)→open(T) gap**. This is internally consistent with the international literature on overnight returns ([Lou-Polk-Skouras 2019](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf); [Knuteson 2020](https://arxiv.org/pdf/2010.01727)) which documents that across 21 major indices (including DAX, FTSE, Nikkei) overnight returns are systematically positive while intraday returns are systematically negative — exactly the pattern our PR #22 result hints at for OMX.

### Tested but inconclusive
- **`vix_2d` (raw VIX 2-day change)** emerged as the single strongest signal in the signal-source comparison (held-out Pearson −0.141, multivariate t=−2.96). PR #10's composite VIX score with deadband *did not* surface this; the level/change distinction matters. This is a clean inadvertent finding from the EU/US analysis, untested as a standalone strategy.
- **The cost model assumes constant % spread regardless of leverage.** Real Nordnet Markets spreads at 10–15× are not yet validated. CLAUDE.md explicitly flags this assumption; it remains the single largest unknown in the cost-wall math.

### Explicitly *not* tested by us, still open
- Multi-day holding horizons (2–5 days). All five tests held intraday or one night. The cost-wall math (see §3.1) is materially different at 3–5 day holds.
- Direct index futures (`OMXS30 futures`) — tick spread ~0.004%, vs ~0.50% on certs. This single change collapses the cost wall by ~125×.
- Volatility regime conditioning (HMM on VIX as meta-filter). Tier 1 #2 in the original menu, never tested.
- Calendar/event effects (FOMC pre-drift, turn-of-month, OPEX).
- Portfolio-of-weak-signals construction. Grinold's IR = IC · √breadth is unused; every test was a single concentrated signal.
- Volatility-as-asset (VIX/VSTOXX futures, options on `^OMX`).
- Markets other than OMX (Nasdaq, individual constituents, futures across regions).

This is a long list of unsearched space. The honest read is **not "the world has no edge" but "we have searched a narrow slice of it"** — specifically: single-signal, 5×-cert, open→close-intraday-on-an-index. Five points inside a small envelope is consistent with that envelope being dead and the rest of the space being unexplored.

---

## 3. Research directions

### 3.1 Multi-day momentum holding (2–5 days)

**Hypothesis.** Index-level *time-series* momentum has documented persistence on horizons longer than one day; capturing 0.5–1% over 3–5 days clears cost better than capturing 0.10% intraday, even after vol decay.

**Economic story.** [Moskowitz-Ooi-Pedersen 2012 "Time Series Momentum"](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf) and [Asness-Moskowitz-Pedersen 2013 "Value and Momentum Everywhere"](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf) establish positive auto-correlation on 1–12 month horizons across equity index futures globally. Short-horizon (1–5 day) momentum is less studied because short-term *reversal* (Jegadeesh 1990) dominates at the individual-stock level — but reversal is a *cross-sectional* phenomenon driven by liquidity provision, not an index-level fact.

**Cost-wall math** (5× cert):
- 1-day round-trip: 0.12% underlying breakeven
- 3-day hold: round-trip 0.12% + financing 0.018% + vol decay ≈ 0.18% = **~0.32% breakeven**
- 5-day hold: ≈ **0.45% breakeven**

A 0.5% 3-day captured move pays. The question is whether such moves are predictable.

**What to test.** TSMOM signal: sign of trailing 20-day return → hold 3 days, 5× cert. Compare to passive baseline. Frozen pre-reg, OOS-split.

**Data/infra.** Existing OMX daily bars. No new ingest.

**Effort.** 4–6 hours (similar to PR #14 build).

**Information value.** HIGH. First multi-day test in our codebase; cleanly resolves whether the open→close limitation was a horizon problem.

**Confidence (real edge): MEDIUM.** Documented at index-level but post-publication erosion is documented ([McLean-Pontiff 2016](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365) showed published anomaly returns fall ~58% out-of-sample). Smaller markets (OMX) may retain more.

---

### 3.2 Single-night gap capture (overnight return premium)

**Hypothesis.** The overnight return premium documented internationally applies to OMX too; we can capture it directly under the ratified narrow no-overnight exception.

**Economic story.** [Lou-Polk-Skouras 2019](https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650) documented that across 14 trading strategies, profits are almost always earned *entirely overnight or entirely intraday* with opposite signs — momentum is overnight; value/profitability/investment are intraday. [Knuteson 2020 "Strikingly Suspicious Overnight and Intraday Returns"](https://arxiv.org/pdf/2010.01727) showed the pattern holds for 21 indices over 32 years. Our PR #22 found train/test sign agreement on the gap (+0.065%); this isn't accidental.

**Cost-wall math.** Same as PR #22's 0.63% round-trip → 0.126% underlying breakeven. The gap edge we observed (+0.065%) is real but sub-breakeven by ~2×. *Two* paths plausibly close that gap: stronger signal selection (5/5 confluence vs 4/5; or filter to high-VIX states; see §3.8), OR migration to futures (§3.6) which collapses the wall by ~50×.

**What to test.** (Already tested in PR #22 at the canonical config. The next test is *not* another gap-capture variant on certs — it is one of: (a) the same trade on OMXS30 futures, §3.6; or (b) the same trade conditioned on a regime filter, §3.8.) **This direction is now coupled to the instrument and regime questions; the gap-capture concept itself is validated as a research direction by PR #22.**

**Confidence (real gross edge): MEDIUM-HIGH** (PR #22 + international literature converging). **Net edge at 5× cert: LOW**. Net edge on futures: not yet measured but mechanically much higher.

---

### 3.3 Calendar and event-driven effects

**Hypothesis.** A small set of high-confidence, well-documented calendar/event windows produces predictable directional drift that survives even on certs because of magnitude.

**Three concrete candidates with literature**:

- **Pre-FOMC announcement drift** ([Lucca-Moench 2015, JoF](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12196)): equity indices drift up ~30–50 bp in the 24 hours before scheduled FOMC announcements. Effect observed in international indices too (DAX, FTSE etc. — they explicitly tested this). Cleanly selective: ~8 FOMC dates/year. *Caveat*: the [2021 follow-up by Brusa-Savor-Wilson](https://ideas.repec.org/a/eee/finlet/v40y2021ics1544612320315956.html) found the drift has weakened post-2015.
- **Turn-of-month** ([McConnell-Xu 2008](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1135217)): found in 31 of 35 countries studied including Sweden. Last 3 + first 3 trading days of the month account for ~most of the equity premium historically. Selective by construction (~6 days × 12 months = 72 days/year, ~28% of trading days).
- **OPEX week effect** ([gex-metrix](https://www.gexmetrix.com/blog/opex-effects); broad evidence): gamma pinning to high-OI strikes during monthly OpEx weeks; mean-reversion to round numbers; post-OpEx pullback. Documented for SPX; mechanism less clean for OMX (smaller options market) but worth quick-check.

**What to test.** Single backtest of "long OMX intraday on FOMC announcement day"; "long OMX last-3 / first-3 days of month". Lookahead-safe is trivial — dates are known years in advance. OOS split same as everything else.

**Data/infra.** Need FOMC calendar (small CSV) + Swedish trading calendar. No new yfinance.

**Effort.** 6–8 hours including the calendar wiring.

**Information value.** HIGH for the FOMC test specifically — cleanest pre-registered test in the menu (8 dates/year, decades of literature backing it).

**Confidence: MEDIUM.** Strong literature; partial decay post-publication; need to verify it transfers to OMX specifically.

---

### 3.4 Cross-market lead-lag (US overnight → OMX open)

**Hypothesis.** The Stockholm open at 09:00 CET imperfectly prices the overnight US session (NYSE closed at 16:00 ET = 22:00 CET prev day; Asian session followed). SPX futures (continuous) and Nikkei (Asia close at ~07:00 CET) provide *fresh* information that should leak into OMX during the first 30–60 minutes of Stockholm trading.

**Economic story.** Lead-lag literature ([surveyed in arxiv 2201.08283](https://arxiv.org/pdf/2201.08283)) consistently finds the more liquid market leads the less liquid by 5–45 minutes. Index futures lead cash by ~5–10 minutes; cross-market the lag is longer. For Stockholm specifically, the SPX overnight session is the most direct comparable — our cross-asset confluence implicitly used this, but at the *daily* horizon. The *intraday* version: on Stockholm open, where SPX-futures overnight move >X bp, OMX opens with a partial gap, and the *remainder* of the move plays out in the first hour of Stockholm trading.

**What to test.** On the 1h `^OMX` bars (already ingested in #17): regress OMX 09:00→10:00 Stockholm bar return on SPX-futures overnight return (close to Stockholm-open). Then test directional trade: if SPX-futures overnight |return| > threshold, take the first Stockholm bar in the same direction. Selective by threshold.

**Data/infra.** Need SPX futures (ES=F on yfinance, or `^SP500-FUT`) at hourly resolution — yfinance has this with same 730-day limit as `^OMX` 1h.

**Effort.** ~1 day (new ticker, 1h ingest, strategy).

**Information value.** MEDIUM-HIGH. Distinct from gap capture (this is *intra-first-hour*, not the full overnight) and from the failed open→close.

**Confidence: MEDIUM.** Mechanism is clean; many retail traders already exploit it; arbitrage may have closed it.

---

### 3.5 Portfolio of weak signals (Grinold breadth)

**Hypothesis.** Multiple weak independent signals combined via inverse-vol or risk-parity weighting produce a portfolio Sharpe materially higher than any individual signal — even when each individual IC is small.

**Economic story.** [Grinold's Fundamental Law](https://blankcapitalresearch.com/learn/grinold-fundamental-law-active-management): IR = IC · √breadth. If we have 10 weakly-correlated signals each with IC = 0.05, the portfolio IR ≈ 0.05 · √10 = 0.16 — meaningfully tradeable. Our current architecture tests one signal at a time at maximum exposure; this is the opposite of breadth.

**What to test.** Take the 5 signals from PR #10 + the 3 from the analysis branch where coefficients had p<0.05 in *either* period + FOMC dummy + TOM dummy. Build an inverse-vol-weighted aggregate score; trade above ±k threshold. Pre-reg the weighting and threshold on 2018-2022; apply blind to 2023-today.

**Data/infra.** All data already ingested.

**Effort.** 2 days (construction + careful OOS to avoid combining the wrong way).

**Information value.** HIGH. Directly tests whether our zero-edge results were a single-signal problem or a market-fact problem. Also produces methodological infrastructure useful for any future strategy.

**Confidence: MEDIUM.** The math is sound; the failure mode is signal *correlation* (signals aren't independent → √breadth overstates). With careful orthogonalization the construction has positive expected value as a test.

---

### 3.6 Switch instruments: OMXS30 index futures

**Hypothesis.** The single most leverage-providing change we can make is *not changing signals* — it is changing instruments. OMXS30 futures collapse the round-trip cost from ~0.50–0.60% (cert spread) to ~0.004% (one tick on the 3400-level index, 12.5 SEK per tick on a 340,000-SEK contract = ~0.4 bp).

**Economic story.** No story needed — this is structural cost arithmetic. Per [Nasdaq Stockholm OMXS30 futures factsheet](https://www.nasdaq.com/docs/2024/12/02/Futures-on-OMXS30-Factsheet): contract = index × SEK 100, tick = 0.125 = SEK 12.5. Margin is ~3–12% of notional. The signal we observe doesn't change; the cost wall it must clear goes from 0.12% to ~0.001% — a **~100× reduction**.

This single change converts *every previously-marginal cert strategy into a potentially tradeable futures strategy.* The cross-asset gap capture (PR #22), which produced +0.065% gap-direction edge but lost net at 5× cert, would *clearly* clear futures cost. The opening-range break (PR #20), which had a trade rate of 86%, would still die on selectivity but the cost-per-trade math becomes much friendlier.

**What to test.** Replicate PR #22's cross-asset gap-capture trade in a futures cost model. Frozen — same signals, same split, only the cost number changes. Then if the math works, surface the instrument-switch decision as its own conversation (different infrastructure, account type, capital, regulatory profile).

**Data/infra.** No new data. Cost-model change only.

**Effort.** Half a day for the cost-model swap; the *operational* shift to live futures trading is its own ~Phase 2 project.

**Information value.** **VERY HIGH.** Likely reveals that several of our "failed" strategies were not signal failures but instrument failures. This re-interprets the entire Phase 0 corpus.

**Confidence: HIGH that the cost-wall math collapses.** MEDIUM that the underlying gap edge is real enough to survive even the lower cost wall (PR #22's +0.065% with sign agreement is the empirical anchor; +0.065% at 0.004% futures cost = clean net edge if the signal persists). But: *we have to actually verify* the futures spread under retail-friendly broker terms; the 0.004% number is the exchange tick, not what a retail broker quotes after their markup.

---

### 3.7 Different markets: Nasdaq, individual stocks, volatility

**Hypothesis.** OMX is small (~30 names), low-vol relative to Nasdaq, and has limited options/futures depth. Edge per cost-unit might be materially higher on Nasdaq-100 (more vol, more catalysts, more derivative liquidity).

**Economic story.** PEAD ([Bernard-Thomas 1990](https://en.wikipedia.org/wiki/Post%E2%80%93earnings-announcement_drift)) is one of finance's most robust anomalies — zero-investment portfolios formed on standardized unexpected earnings (SUE) deciles earned ~8–9% per quarter (35% annualized before transaction costs). But: *the anomaly is concentrated in small-cap stocks*, and CLAUDE.md explicitly excludes single-stock certs as out-of-scope. The clean retail-accessible market with strong documented anomalies and accessible derivatives is **`^NDX`** (Nasdaq 100) — same cert infrastructure on Nordnet, much larger options market, and many of the same documented effects.

**What to test.** Repeat the cross-asset gap-capture (PR #22 design) on `^NDX` certs. Same cost model, same split discipline. Or test PEAD on the top 5–10 OMX constituents (Atlas Copco, Volvo, etc.) via single-name certs — but this expands instrument scope materially.

**Data/infra.** `^NDX` already ingested. Single-name OMX constituents would be new ingest.

**Effort.** Cross-asset on NDX: 1 day. PEAD on constituents: 1–2 weeks (data complexity, earnings calendars, idiosyncratic risk modeling).

**Information value.** MEDIUM for NDX gap-capture replica (just confirms market-specific or generic). HIGH for PEAD if we accept single-name scope.

**Confidence: MEDIUM** on NDX (same edge architecture, different microstructure). **LOW-MEDIUM** on PEAD with retail constraints (anomaly is real but heavily arbitraged in liquid US stocks; OMX constituents may retain more but transaction friction is higher).

---

### 3.8 Volatility regime conditioning (HMM as meta-filter)

**Hypothesis.** Strategies that fail averaged-over-all-regimes work in specific regimes. A VIX-state HMM (low-vol / medium-vol / high-vol) used as a meta-filter — only trade when the current regime is favorable — could rescue strategies whose signal is real but blurred by regime-mixing.

**Economic story.** [QuantifiedStrategies review of HMM regimes](https://www.quantifiedstrategies.com/hidden-markov-model-market-regimes-how-hmm-detects-market-regimes-in-trading-strategies/) and [practical implementations](https://blog.quantinsti.com/regime-adaptive-trading-python/) consistently report that simple HMM-based *standalone* strategies underperform buy-and-hold, but HMM as a *meta-filter* on top of another signal can eliminate losing trades and improve Sharpe. This is Tier 1 #2 in our original edge menu and was never tested.

Specifically relevant to our gap result: PR #22 found train n=200 with t=+2.10 on LONG signals and t=+0.53 on SHORT signals (huge per-direction variance). If a regime filter could identify *which* direction works in which state, the pooled edge might rise substantially.

**What to test.** Fit a 2-state HMM on OMX daily returns + VIX level. Re-evaluate each prior strategy *conditional* on regime. Pre-reg the regime fit on 2018-2022; apply blind to 2023-today. Critical lookahead-safety: regime probabilities must use trailing data only.

**Data/infra.** Existing. Need `hmmlearn` Python package or implement Baum-Welch (avoid the second; use `hmmlearn` as a dev dep).

**Effort.** 2 days (HMM fitting + careful lookahead-safe regime inference + re-running 5 strategies).

**Information value.** HIGH if any strategy comes back from the dead under a regime filter; MEDIUM otherwise. Either way it validates or kills the meta-filter approach generically.

**Confidence: MEDIUM-LOW** that this resurrects net edge on its own. **HIGH** that it produces a clean answer.

---

### 3.9 Volatility risk premium (short volatility / options selling)

**Hypothesis.** Selling index options harvests the well-documented volatility risk premium — implied vol systematically exceeds realized vol; the seller of options collects the difference as a premium for bearing vol-of-vol exposure.

**Economic story.** [Quantpedia "Volatility Risk Premium Effect"](https://quantpedia.com/strategies/volatility-risk-premium-effect) summarizes a large literature: SPX 1-month at-the-money implied vol exceeds subsequent realized vol on ~80% of months historically; the difference (the VRP) compensates option sellers for tail risk. [Carr-Wu studies and FRBNY's "Equity Volatility Term Premia"](https://libertystreeteconomics.newyorkfed.org/2021/02/equity-volatility-term-premia/) document this rigorously.

**The catch.** Selling options has *unbounded tail risk*; one bad month (Feb 2018 vol-magedon, March 2020) can erase years of premium. Retail implementation requires margin reserves and discipline. This is structurally the most different proposal in the menu — not directional, not predictive, just "collect insurance premiums net of claims paid".

**What to test.** *Not* directly tradeable for us in Phase 0 — Nordnet's retail offering for OMX options is limited; would need to use VSTOXX or US options via a different brokerage. Long-term direction worth surfacing but not the next test.

**Effort.** Substantial — different infrastructure, different risk model, different account type.

**Information value.** LOW-MEDIUM for our specific Phase 0/1 trajectory.

**Confidence (real gross edge): HIGH** (the VRP is robust and persistent). **Confidence (accessible to us with current setup): LOW.**

---

## 4. Constraints we should reconsider

The hard rules in CLAUDE.md were set *before* we had Phase 0 evidence. Four deserve formal reconsideration now.

### No-overnight rule — RECONSIDER (already partially done)
**Status: partially relaxed via PR #22.** The case for a one-night exception is now empirically grounded — overnight return premium is internationally documented (Lou-Polk-Skouras 2019; Knuteson 2020); PR #22 confirmed sign-agreement train/test for OMX. The remaining question is whether to allow 2–5 day holds for momentum (§3.1). **Recommendation: extend the exception list to a second specific strategy** — pre-registered TSMOM 3-day hold — *if* §3.1's pre-reg test motivates it. Don't open the rule generically.

### 5× default leverage — RECONSIDER (and lock to "5× cert OR equivalent")
The 5× default makes sense at the *cert* cost wall. But the cost wall depends on instrument and leverage. **Recommendation: re-frame the rule as "5× exposure" agnostic to instrument**, validated separately for certs (current ~0.6% round-trip), futures (~0.004%), and any other instrument we add. The "leverage" number alone is the wrong unit; the relevant unit is "exposure per cost-unit", i.e., 1/breakeven_underlying.

### OMX-only focus — KEEP, with explicit exception process
OMX-only kept the search disciplined in Phase 0. The four-data-point gap-arbitrage finding generalises *across markets* (literature on DAX/FTSE/Nikkei confirms this) — so we don't need to widen the market to validate the broader thesis. We *do* need to widen if any specific edge is shown to be OMX-pathological (e.g., if §3.7's NDX replica works and OMX doesn't). **Recommendation: KEEP the OMX-only default but add explicit per-test override** ("this test runs on NDX because we hypothesise OMX-specific microstructure friction; here is the test").

### Same-day open→close model — PROVISIONALLY DROP
This was the *implicit* envelope underneath every Phase 0 test. Five OOS failures inside it warrant explicit replacement. **Recommendation: PROVISIONALLY DROP** — replace with "trade horizon is part of every new strategy's pre-registration; default is no longer same-day open→close." This isn't deleting the model; it's stopping its inertial application to every new test.

---

## 5. Ranked recommendation

Of the nine directions in §3, the two I would actually pursue next are:

### Rank 1 — §3.6 Switch to OMXS30 futures cost model + replicate PR #22

**Why first.** This is the single highest information-per-hour test in the menu. It does not require a new strategy, new signal, or new architecture. It changes a single number (round-trip cost from 0.63% to ~0.01% on futures) in the existing PR #22 codebase and reruns the OOS. The math is mechanical; the result will tell us in 4–6 hours of work whether five "failed" strategies were really *cost-bound* (in which case the entire Phase 0 corpus reinterprets) or *signal-bound* (in which case we've cleanly established that bigger-picture finding too). Even a negative result here — "futures cost still doesn't save it" — is highly valuable because it rules out the instrument-switch hypothesis with one shot.

**Honest prior on finding edge.** ~35%. The +0.065% gap edge is real (sign agreement train/test); the cost wall is mechanically obliterated; the only failure mode is that retail futures access has higher effective spreads than the exchange tick suggests. Worth a 4–6 hour test.

**What we give up.** Some Phase 2 prep work. Nothing irrecoverable.

### Rank 2 — §3.5 Portfolio of weak signals (Grinold breadth)

**Why second (parallel-able).** Five OOS failures using *one signal at a time* is a methodological hint, not just a market fact. Grinold's IR = IC · √breadth gives a quantitative reason to believe that *combining* signals — orthogonalised and inverse-vol weighted — produces materially better risk-adjusted edge than the best individual signal. We already have ~10 signals (US + EU + vix_2d) with measured IC ≈ 0.03–0.10. The construction is well-understood; the failure mode is signal correlation; with careful orthogonalisation the test has clear positive information value. **Crucially, this can be done in parallel with Rank 1** — it touches different modules and gives an independent answer about the architecture.

**Honest prior on finding edge.** ~25% at 5× cert cost wall (the breadth helps but the cost wall is still 0.12%). **~55% at futures cost wall** (combining Rank 1 + Rank 2). The combination is the highest-EV path I see.

**What we give up.** ~2 days of work. Nothing irrecoverable.

### What I'd explicitly defer

- **§3.1 Multi-day momentum** is high-quality but should wait for Rank 1's result. If futures collapse the cost wall, the 1-day momentum we *already see in PR #22* is enough; multi-day adds risk (vol decay) without proportional return.
- **§3.3 Calendar/event** is high information value but lower expected-edge magnitude (FOMC drift is 30–50 bp in US data, partially decayed post-2015). Cleanly testable, but smaller upside per unit work.
- **§3.7 NDX replica** is the natural fast-follow to Rank 1 if futures work for OMX — adds robustness without much new infrastructure.
- **§3.8 HMM regime** is structurally important but high-effort relative to expected-edge. Defer until we've decided whether the strategies are worth conditioning at all.
- **§3.9 Options/VRP** is the most different and most retail-inaccessible at our current setup; correct medium-term consideration but not the next test.

---

## 6. Intellectual honesty: what if there's no accessible edge?

The most uncomfortable possibility to engage with honestly: **what if, for the specific configuration of "Swedish retail trader with daily-bar / 1-h data, leveraged certificates with 0.5–0.6% round-trip cost, no overnight holds, no institutional infrastructure", *there is simply no net-tradeable edge*?**

There is a serious case for this. Five OOS failures is not nothing. The signal magnitudes we observe (+0.065% best-case gap edge, 3% best-case multivariate R²) are consistent with what efficient-market theory predicts for a market that has been continuously traded by institutional participants for decades. [McLean-Pontiff (2016)](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365) showed published anomaly returns decay by 58% out-of-sample post-publication; the anomalies we are searching for *are precisely the ones most likely to have been arbitraged*. Lou-Polk-Skouras's "overnight premium" — the strongest empirical pattern we have to grab onto — is a *risk* premium for bearing overnight gap exposure, not a *free lunch*; the compensation may be exactly right for the risk in the long run.

The instrument-switch hypothesis (Rank 1 above) is the single test most likely to falsify this pessimistic view. If futures collapse the cost wall and PR #22's gap edge becomes net-positive, that's a clean *negative result against the pessimistic view*. If it doesn't, the pessimistic view gains substantial support. Either way, the cost is low.

If the pessimistic view turns out to be right, the honest answer for an individual retail trader is not "keep searching harder" — it is **passive long-only OMX exposure or a global index fund**, supplemented at most by very-low-frequency rebalancing or known long-horizon premia (value, quality, profitability). The willingness to engage with this answer is what separates an honest research process from one that confabulates edge to justify continued effort. We should be willing to write that conclusion in plain language if the evidence forces it.

The narrow space where I think there *is* probably accessible edge for a careful retail trader is exactly the two top recommendations: **(a) reducing cost friction via better instruments**, where the math is mechanical and the institutional players already exploit this; and **(b) combining multiple weak signals**, where Grinold's law gives a quantitative reason to believe in a small but real edge that no single-signal test can detect. Both are testable cheaply. Neither requires us to discover something the academic literature has missed — only to apply known constructions to our specific market.

If after Rank 1 + Rank 2 there is still no net edge, the honest recommendation is to declare the search complete, document the result, and either (i) accept passive exposure as the answer, or (ii) escalate to a structurally different commitment: Phase 1.5 with live broker integration and real-spread measurement, accepting the operational cost in exchange for the chance that our cost model has been systematically pessimistic. That is a *capital* and *time* decision, not a research decision.
