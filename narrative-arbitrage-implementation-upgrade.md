# Narrative Arbitrage — Implementation Document: RL Upgrade Path

Sequenced by leverage (biggest impact on results/story first), not by difficulty. Each phase assumes the current repo (Phases 0–6 done, Phase 7 code-complete) as the starting point. Stop at any phase and the project is still coherent — later phases are "reach," not required.

---

## Upgrade 0 — Resolve the Known Issues First (do this before anything else)

**Why first:** building risk-sensitive RL on top of an unresolved staleness bug and an unaudited reward function just buries the problems deeper.

**Dev prompt:**
```
In the narrative-arbitrage repo:

1. Resolve the DOGE-only staleness (docs/CURRENT_STATUS.md §04, Option A):
   - Update final_report.md §5a to note all 5 assets now have trained
     checkpoints as of 2026-09-15
   - Update the 6 failing tests that hard-assert DOGE-only to instead check
     has_trained_policy() dynamically for all 5 assets
   - Run the full test suite and confirm 90/90 pass

2. Run src/eval/reward_hacking_check.py against the AMC and GME checkpoints
   specifically (their single-window Sharpe of -2.03 and -3.82 is the
   project's own stated reason to be suspicious of them). Report: is the
   agent collapsing to a degenerate policy (always-short, always-max-position)
   on these two assets, or is it a genuine (if bad) learned policy? Document
   the finding in docs/reward_hacking_findings.md before any further changes
   are made to these two assets' training.

3. Source an alternative to Reddit for comment-acceleration/sentiment data,
   since Reddit stopped issuing self-serve API credentials as of 2026-09.
   Evaluate StockTwits API (free, finance-specific, built for this exact use
   case) and/or 4chan's /biz/ API (free, no auth) as replacements. Implement
   whichever is viable in src/data_pipeline/ following the same interface as
   the existing wikipedia.py/trends.py fetchers, and backfill the missing
   comment_count/sentiment/novelty-score columns in data/processed/.
```

---

## Upgrade 1 — Risk-Sensitive Objective (highest leverage, fixes AMC/GME directly)

**Goal:** Replace (or augment) the plain expected-reward PPO objective with a distributional or CVaR-aware objective, so the agent is explicitly penalized for tail risk rather than only optimizing the mean.

**Dev prompt:**
```
Implement a risk-sensitive variant of the current PPO training in
src/agents/:

1. src/agents/distributional_critic.py — replace the standard scalar-value
   critic with a quantile-regression critic (IQN or QR-DQN style: predict
   N quantiles of the return distribution instead of a single expected
   value). Use this to compute a CVaR-adjusted advantage estimate: instead
   of standard GAE on the expected value, weight the advantage using the
   lower-tail quantiles (e.g. bottom 10-20%) so the policy gradient
   explicitly penalizes downside risk, not just average return.

2. Add a config flag (risk_sensitivity: "none" | "cvar") to the training
   script from Phase 4a/4b so both the original expected-return PPO and the
   new risk-sensitive PPO can be trained and compared on identical data/seeds.

3. Retrain AMC and GME specifically (the two worst performers) under both
   objectives. Report walk-forward Sharpe, Sortino, and max drawdown for
   both versions side by side in docs/risk_sensitivity_ablation.md. The
   expected outcome to test for: does the CVaR-aware agent trade less
   aggressively / hold smaller positions during high-uncertainty periods,
   and does that materially reduce the catastrophic single-window losses
   without destroying upside on the assets that were already fine (DOGE)?

4. Extend to all 5 assets once the AMC/GME pilot confirms the approach works
   and doesn't harm the already-decent assets.
```

---

## Upgrade 2 — Attention-Based State Encoder

**Goal:** Replace the flat MLP feature encoder with a lightweight temporal-attention (Transformer-style) encoder over the sequence of attention-derivative features, so the policy can learn patterns in *how acceleration unfolds over time*, not just single-timestep snapshots.

**Dev prompt:**
```
In src/env/ and src/agents/, extend the observation to include a short
lookback window (e.g. last 10-20 timesteps) of attention-derivative features
per asset instead of a single-timestep vector. Implement a custom feature
extractor for stable-baselines3:

1. src/agents/attention_encoder.py — a small Transformer encoder (2-4 layers,
   4 heads is plenty at this data scale) or a simpler temporal-attention
   pooling layer, that takes the lookback window as input and produces a
   fixed-size embedding. Register this as a custom
   BaseFeaturesExtractor (SB3's extension point) so it plugs into PPO
   directly.

2. Run three configurations on identical data/seeds/reward:
   (a) original flat MLP (baseline, already have this)
   (b) LSTM/RecurrentPPO (this is also Upgrade 3, do them together)
   (c) Transformer-based encoder (this upgrade)

3. Compare walk-forward Sharpe/Sortino across all three per asset. Report
   results in docs/architecture_ablation.md, structured as a proper ablation
   table: architecture x asset x metric. This table is the centerpiece of
   the "which architecture actually helps" finding.
```

---

## Upgrade 3 — RecurrentPPO, Completed as a Real Ablation

**Goal:** Finish the phase that was originally scoped and cut — but frame it as a controlled comparison against Upgrade 2, not a standalone add-on.

**Dev prompt:**
```
Using sb3-contrib's RecurrentPPO, train the same 5-asset basket with an LSTM
policy on the same lookback-window observation from Upgrade 2, same reward,
same seeds. This slots directly into the architecture_ablation.md table from
Upgrade 2 as the third row (MLP vs LSTM-PPO vs Transformer-PPO).

Explicitly test and report: does sequential memory (LSTM) capture what the
Transformer's attention captures, more, or less? This was the original
"sequential memory should help" hypothesis from the project's problem
statement — now it gets a real answer instead of being a scope cut.
```

---

## Upgrade 4 — Hyperparameter Search & Multi-Seed Reporting

**Goal:** Replace the single seed=42 result with a defensible statistical claim.

**Dev prompt:**
```
Add src/agents/hparam_search.py using Optuna: sweep learning rate, entropy
coefficient, GAE lambda, and network width for the best-performing
architecture from Upgrades 1-3, on one representative asset (DOGE, since it's
the most stable). Use a small budget (20-30 trials) — this doesn't need to be
exhaustive, just defensible.

Then retrain the final chosen configuration across 5 seeds per asset. Report
all backtest metrics as mean ± std across seeds, not single numbers, in the
final results table. Update docs/final_report.md's results section to use
mean ± std throughout instead of point estimates.
```

---

## Upgrade 5 — Joint Multi-Asset Portfolio Policy (reach item)

**Goal:** Move from 5 independent single-asset policies to one policy that allocates across the whole basket jointly, so it can learn cross-asset correlation and diversification.

**Dev prompt:**
```
Extend src/env/trading_env.py into a portfolio-level environment: observation
is the concatenated (or attention-pooled, reusing Upgrade 2's encoder) state
across all 5 assets, action is a vector of position sizes (one per asset,
e.g. via a softmax-constrained allocation or independent Box actions with a
leverage cap). Reward is portfolio-level risk-adjusted return, not
per-asset.

Train this jointly and compare portfolio-level walk-forward Sharpe against
the sum of the 5 independent single-asset policies from earlier phases —
the interesting question is whether joint training reduces drawdown through
implicit diversification even if it doesn't improve raw return.
```

---

## Upgrade 6 — Offline Pretraining / Decision Transformer Warm-Start (reach item)

**Goal:** Address the single-window vs. walk-forward instability (e.g. DOGE's 3.95 → 1.71 swing) by pretraining on existing trajectory data before online PPO fine-tuning.

**Dev prompt:**
```
Using the trajectories already generated by the momentum and supervised
baselines (Phase 2) plus the walk-forward backtest runs (Phase 5) as offline
data, implement either:
(a) behavior cloning pretraining: initialize the PPO policy's weights by
    supervised-learning it to mimic the better-performing baseline's actions,
    then fine-tune with PPO as normal, or
(b) a minimal Decision Transformer: train a sequence model to predict actions
    conditioned on return-to-go, using the same offline trajectories,
    evaluate it directly as an alternative to PPO altogether.

Compare walk-forward stability (variance of Sharpe across rolling windows)
between the from-scratch PPO baseline and the pretrained/DT version. The
hypothesis to test: does pretraining reduce the single-window vs
walk-forward disagreement seen in the current results?
```

---

## Suggested Stopping Points

If time is limited, in order of what to complete:

1. **Upgrade 0** (fix known issues) — always do this, it's cheap and protects credibility
2. **Upgrade 1** (risk-sensitive objective) — highest leverage, directly explains and fixes your worst result
3. **Upgrades 2+3 together** (architecture ablation) — strong "which design choice actually mattered" narrative, moderate effort
4. **Upgrade 4** (hyperparameter search) — cheap statistical rigor, do this regardless of what else you attempt
5. **Upgrades 5–6** — only if time allows; fine to write up as "future work" in the final report otherwise

## Final Report Restructuring

Once any of the above is complete, `final_report.md` should be restructured around this arc:
1. Original hypothesis and initial PPO results (already have this)
2. The failure: AMC/GME catastrophic single-asset losses under expected-return optimization
3. Diagnosis: reward hacking audit (Upgrade 0) ruled out / identified as the cause
4. The fix: risk-sensitive objective (Upgrade 1) and its measured effect
5. Architecture ablation: what representation of "attention" actually helps a policy learn (Upgrades 2–3)
6. Final results with proper statistical reporting (Upgrade 4)
7. Future work: portfolio-level and offline-pretrained extensions (Upgrades 5–6)

This structure turns your current negative result into the spine of the whole write-up rather than an embarrassing footnote — which is exactly what makes it read as research rather than a tutorial project.
