# Current Status

*Rewritten 2026-09-20 to consolidate a long chain of dated addenda into one
coherent, non-contradictory document. Nothing below is invented — every
number is sourced from the doc named next to it. Not a git repo (no
`.git/`), so ordering claims come from filesystem mtimes and session
history, not commit history. Superseded by nothing yet — if this file and
another doc ever disagree, this file is the one to trust and the other one
needs a fix.*

## Project summary

PPO-based RL trading system that trades a basket of meme stocks (AMC,
GME), small-cap crypto (DOGE, PEPE, SHIB), and 2 NFT collections
(price-less, feature-only) using **only** social-attention-derivative
features (Wikipedia pageviews/edits, Google Trends search interest,
Hacker News volume, a live-only 4chan signal) as state — price is used
solely for reward computation, never as an input feature. Evaluated with
leakage-audited walk-forward backtesting against a momentum baseline and
a same-features supervised (XGBoost) baseline. Beyond the original single
architecture, this project now includes a full RL upgrade path — CVaR
objective, Transformer encoder, RecurrentPPO, hyperparameter search,
seed ensembling, a joint multi-asset policy, and behavior-cloning
pretraining — each walk-forward re-validated where noted, single-window
only where not.

**Test suite: 140/140 passing** (`pytest -q`, ~21s, no network calls).

## The headline finding

Every RL upgrade built on top of the original PPO agent (CVaR objective,
Transformer encoder, LSTM/RecurrentPPO) **looked like a clear win when
evaluated on a single validation window, and lost to the original
unmodified PPO once walk-forward re-validated** —
`docs/walk_forward_upgrade_validation.md`:

| Method | Walk-forward Sharpe (5-asset basket avg) |
|---|---:|
| Momentum (baseline) | 1.119 |
| **Original PPO (plain MLP)** | **0.712** |
| Transformer encoder | 0.512 |
| RecurrentPPO (LSTM) | 0.451 |
| CVaR-objective PPO | 0.110 |
| Supervised (XGBoost) baseline | 0.065 |

The one confirmed exception is **LSTM specifically on GME** (the basket's
worst-performing asset, −1.00 → +0.89 mean across 3 seeds) — real but not
fully seed-robust on its own (one of three seeds came back negative). A
**3-member seed ensemble resolves this cleanly**: Sharpe 2.139 on GME,
beating every individual seed tried, including the best one
(`src/agents/ensemble.py`). This is the one piece of the entire upgrade
path that unambiguously earns its extra complexity.

## RL upgrade path — what's done

| # | Upgrade | Status | Where |
|---|---|---|---|
| 0 | Known-issues resolution (staleness fix, reward-hacking audit, Reddit-alternative eval) | **Done** | `docs/reward_hacking_findings.md`, `src/data_pipeline/fourchan_biz.py` |
| 1 | Risk-sensitive CVaR objective | **Done, walk-forward validated — underperforms** | `src/agents/distributional_critic.py`, `docs/risk_sensitivity_ablation.md` |
| 2+3 | Transformer encoder + RecurrentPPO architecture ablation | **Done, walk-forward validated — LSTM wins on GME only** | `src/agents/attention_encoder.py`, `docs/architecture_ablation.md` |
| 4 | Hyperparameter search + multi-seed reporting | **Done** — sweep on DOGE, multi-seed extended to all 5 assets | `src/agents/hparam_search.py`, `docs/hparam_search_results.md`, `docs/multiseed_report.md` |
| — | Walk-forward re-validation of 1-3 | **Done** — the capstone finding above | `docs/walk_forward_upgrade_validation.md` |
| — | Seed ensemble (fix for the seed-variance problem) | **Done on GME, not yet basket-wide** | `src/agents/ensemble.py` |
| 5 | Joint multi-asset portfolio policy | **Done, single-window only** — discovered real cross-asset structure unprompted | `src/env/multi_asset_env.py`, `src/agents/train_portfolio_ppo.py`, `docs/joint_portfolio_validation.md` |
| 6 | Offline pretraining (scoped to behavior cloning, not a full Decision Transformer) | **Built and unit-tested, not yet evaluated at scale** | `src/agents/behavior_cloning.py`, `--rl-variant bc_pretrained` in `walk_forward.py` |
| — | Attribution extended beyond DOGE to all 5 assets | **Done** | `docs/attribution_report_all_assets.md` |
| — | Live loop actually run against real APIs | **Done once** (continuous operation needs cron — see below) | `docs/live_loop_first_run.md` |
| — | Interactive frontend (`/research` page, real charts, README) | **Done, first pass — a second interactivity pass is in progress** | `frontend/app/research/page.tsx`, `README.md` |

## Joint portfolio policy — the other interesting finding

`MultiAssetTradingEnv` trains one policy across all 5 assets jointly
(270-day common-calendar window). Its per-asset action correlations
cluster exactly along asset-class lines **with no explicit label ever
telling it which assets are crypto**: DOGE/PEPE/SHIB (the 3 crypto
assets) correlate 0.64–0.70 with each other; GME/PEPE is −0.21. Same
"better Sharpe/Sortino, worse cumulative return/drawdown" pattern as
CVaR — single-window only, not walk-forward validated (`docs/
joint_portfolio_validation.md`).

## Data pipeline — honest gaps

- **Reddit**: fully implemented (`src/data_pipeline/reddit.py`), never
  produced data — Reddit stopped issuing self-serve API credentials as of
  2026-09. `comment_count`/`sentiment`/`novelty_score` are absent from
  every processed feature file, permanently (StockTwits and 4chan were
  both evaluated as replacements; 4chan works but has no historical
  archive, so it's live-only — see `docs/feature_dictionary.md`).
- **NFTs** (`boredapeyachtclub`, `cryptopunks`): have processed attention
  features, no price data — excluded from training/backtesting/live
  trading, included only in the cross-platform lead-lag analysis (needs
  no price).
- **4chan `/biz/`**: works live, no auth needed, wired into the live
  scheduler as a separate signal (`live_biz_signals` table) — deliberately
  NOT merged into the PPO observation, since that would silently change
  trained checkpoints' expected input dimensionality (see
  `src/realtime/db.py`'s docstring).

## Live loop

Actually run end-to-end against real APIs for the first time on
2026-09-20 (`docs/live_loop_first_run.md`) — `data/live/live.db` is
populated in every table, and the backend/frontend were verified to
render it. **That first run found and fixed a genuine crash bug**:
`infer.py` didn't convert JSON `null` (Wikipedia pageviews are always
unavailable for "today" — a normal 1-2 day API lag, not an edge case) to
`np.nan`, so every live decision would have crashed. Fixed and
regression-tested. Continuous multi-day operation (the original Phase 7
ask) needs either a long-running foreground process or a cron job — not
something any single session can provide; exact commands are in that doc.

## Tier 2 pass (2026-09-20, later same day) — causal validation, Pareto frontier, uncertainty sizing

Three more independent lines of evidence added on top of everything above:

1. **Causal validation** (`src/eval/causal_validation.py`,
   `docs/causal_validation.md`) — transfer entropy + Granger causality,
   with proper multiple-comparison correction (85 simultaneous tests
   across the basket), between every attention feature and next-day
   return. **Transfer entropy finds ZERO pairs that survive FDR
   correction** — no robust nonlinear lead-lag evidence anywhere in this
   basket at this sample size. Granger causality (linear) finds 8
   survivors, concentrated on GME — and GME's `pageviews_velocity` is
   independently BOTH one of those causal survivors AND the RL policy's
   #2 most-attributed feature (`docs/attribution_report_GME.md`), a real
   convergence between two independent methods. This is the closest thing
   in the whole project to positive causal evidence, and it's specific to
   GME — the same asset where the LSTM ensemble found its biggest win.
2. **Pareto frontier** (`src/eval/pareto_frontier.py`,
   `docs/pareto_frontier.md`) — swept CVaR's own `cvar_alpha` (0.9→0.1),
   walk-forward, on GME and AMC. **Asset-dependent, not one clean
   tradeoff**: for GME, lower alpha improves Sharpe AND drawdown
   simultaneously (not really a tradeoff at all); for AMC, there's a real
   Sharpe-maximizing peak around alpha=0.5, but no alpha value beats
   AMC's original (non-CVaR) PPO Sharpe. Reframes the CVaR "failure" as
   real and asset-specific, not a single verdict.
3. **Uncertainty-aware position sizing**
   (`src/agents/uncertainty_critic.py`,
   `docs/uncertainty_sizing_validation.md`) — scaling a base policy's
   action by an independent critic ensemble's value disagreement. Mixed:
   genuine improvement on both Sharpe and drawdown for GME, but worse
   Sharpe (despite better drawdown) for AMC — a third independent
   confirmation of the "risk techniques trade Sharpe for drawdown, not a
   free lunch" pattern already seen with CVaR and the joint portfolio
   policy.
4. **Seed ensemble confirmed on a second asset**: AMC, Sharpe 1.033 —
   beats every other variant tried for AMC, including original PPO. Now
   2-for-2 (GME, AMC), no longer a GME-specific curiosity.

Test suite: 154/154 passing.

**Explicitly deferred, not built — a GNN cross-asset policy and a
regime-switching hierarchical policy.** Both were part of the same Tier 2
plan; skipped by deliberate priority choice (the plan's own stated
priority order put them 4th and last respectively) in favor of finishing
causal validation, Pareto frontier, and uncertainty sizing first, then
shifting remaining effort to the frontend interactivity pass requested
alongside this plan. The GNN specifically is lower-urgency than it might
seem: the flat joint portfolio policy already found the crypto-cluster
structure a GNN would formalize (see above), so the marginal value of
adding explicit graph structure is a real but second-order question, not
an unanswered one.

## What's still genuinely open

In rough priority order:

1. **Extend the seed ensemble to DOGE/PEPE/SHIB** — 2 of 5 done (GME,
   AMC), both clean wins; mechanical to repeat for the remaining 3.
2. **Sweep the uncertainty-sizing `k` coefficient** the way Phase 3 swept
   `cvar_alpha` — only one untuned value (k=2.0) tried so far.
3. **Evaluate `bc_pretrained` at scale** — implemented and unit-tested,
   never run through the real walk-forward harness.
4. **Walk-forward validate the joint portfolio policy** — currently
   single-window only; the 270-day common-calendar window may be too
   short to subdivide meaningfully, an open design question.
5. **Walk-forward validate the tuned hyperparameter config** from the
   Optuna sweep.
6. **GNN cross-asset policy and regime-switching hierarchical policy** —
   deliberately deferred, see above.
7. **Live attention-signal charts with polling/animation and an agent
   decision-strip component** (frontend plan §2-3) — not yet built; the
   design system (§1) and the two data-showcase pieces most tied to this
   session's actual research (§4 Pareto scatter, §6 correlation network,
   plus §5 architecture bars) were prioritized instead, since they surface
   real findings rather than restyle existing views. Animated polish (§7)
   also not attempted.
8. Reddit credentials remain a genuinely external, unresolvable blocker.

## Frontend interactivity pass (2026-09-20, same day)

Design system (§1 of the frontend plan) plus the highest-value data
visualizations (§4-6) are done:

- **Design system**: dark-by-default (not `prefers-color-scheme`-only),
  a 4-color semantic palette (`--color-gain`/`loss`/`attention-signal`/
  `agent-action`/`price-reward`, `app/globals.css`) reused by every chart
  via a shared theme config (`lib/chartTheme.ts`), JetBrains Mono for all
  numeric displays.
- **`/lab` page** (`frontend/app/lab/page.tsx`), three new components, all
  driven by this session's real Tier 2 data (nothing invented client-side):
  - `CorrelationNetwork.tsx` — a small hand-rolled force-directed layout
    (no external graph library needed for 5 nodes) rendering the joint
    portfolio policy's action correlations. Verified in a real browser:
    the layout visibly clusters DOGE/PEPE/SHIB (crypto) apart from
    GME/AMC (stocks), with the GME-PEPE negative correlation rendered in
    red — the finding demonstrates itself, not just a caption claiming it.
    Nodes are clickable, navigating to that asset's page.
  - `ParetoScatter.tsx` — the `cvar_alpha` sweep from Phase 3, with
    original-PPO/CVaR-α=0.2 reference points marked distinctly (◆).
  - `ArchitectureAblationChart.tsx` — grouped bars, every architecture ×
    asset, walk-forward Sharpe, with a real error bar on the ONE cell
    with actual multi-seed walk-forward data (GME/LSTM) and no fabricated
    error bars elsewhere.
- Home page redesigned with a hero section and live-fetched asset summary;
  existing `FeatureChart.tsx` recolored to the new semantic palette for
  consistency across old and new pages.
- All verified rendering correctly in a real browser (not just "compiles"
  — actually viewed, scrolled, and clicked through).

## Verification notes

- Every number in `docs/final_report.md` was cross-checked against the
  specific report that generated it (backtest, attribution, lead-lag) at
  the time each section was written — see that document's own citations.
- `backend/main.py` and `frontend/lib/api.ts` are wired to the same
  6-route contract; verified working end-to-end in a real browser against
  both replay and live data (2026-09-20).
- No `TODO`/`FIXME`/`NotImplementedError`/stub function bodies anywhere
  in `src/` or `backend/`.

## Where to actually start reading

`docs/walk_forward_upgrade_validation.md` (the capstone finding) →
`docs/final_report.md` (the full write-up) → everything else in `docs/`
is a focused report on one specific piece, linked from those two.
