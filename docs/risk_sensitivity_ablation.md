# Risk-Sensitivity Ablation — Expected-Return PPO vs CVaR-PPO

> **⚠ UPDATE (2026-09-20): this document's headline finding does NOT
> survive walk-forward re-validation.** Walk-forward basket-average Sharpe
> for CVaR is 0.110 — the worst of every RL variant tried, well below the
> original plain PPO's 0.712 — reversing the single-window claim below that
> CVaR improves Sharpe/Sortino on 4 of 5 assets. See
> `docs/walk_forward_upgrade_validation.md` for the full re-validation and
> what it means. Everything below is preserved as an accurate record of the
> single-window result and its own reasoning, but should not be read as
> evidence CVaR is a good idea for this problem — walk-forward says the
> opposite.

Upgrade 1 of `narrative-arbitrage-implementation-upgrade.md`. Implements and
evaluates `src/agents/distributional_critic.py`'s CVaR-aware PPO
(`CVaRPPO` + `QuantileCriticPolicy`) against the original expected-return
PPO, across all 5 basket assets that have a real checkpoint (`docs/
reward_hacking_findings.md` cleared AMC/GME of reward hacking first, per
Upgrade 0's ordering — this ablation was only run after that).

**Mechanism, briefly** (full detail in `distributional_critic.py`'s module
docstring): the critic predicts 32 quantiles of the return distribution
instead of a single expected value; SB3's own GAE/rollout code (unmodified)
is fed `CVaR_0.2` (mean of the bottom 20% of predicted quantiles) as "the
value" wherever it would normally use the mean — making the advantage
estimate, and therefore the policy gradient, pessimistic/tail-aware rather
than expected-value-only. The critic itself is fit via quantile-Huber
regression against the same scalar GAE-lambda return target, not MSE. The
REWARD function is unchanged (still vol-normalized PnL, `NarrativeTradingEnv`
unmodified) -- only the critic and the value baseline used inside GAE
changed.

**Method:** same protocol as the original Phase 4a runs being compared
against -- 100,000 timesteps, seed 42, identical train/validation split per
asset (`src/agents/common.get_train_val_split`), single validation window
(not walk-forward -- see "Scope note" below). `--risk-sensitivity cvar
--cvar-alpha 0.2 --n-quantiles 32`, saved to `models_cvar/ppo_<asset>/`
(kept separate from `models/ppo_<asset>/`, the original "none" checkpoints,
which were NOT retrained -- reused as-is for a byte-for-byte fair baseline).

## Results: all 5 assets, single validation window

| Asset | Objective | Sharpe | Sortino | Max DD | Cum. Return |
|---|---|---:|---:|---:|---:|
| AMC | none (baseline) | −2.030 | −2.456 | −0.911 | −0.908 |
| AMC | **cvar** | **−0.927** | **−0.827** | −1.161 | −1.101 |
| GME | none (baseline) | −3.816 | −3.437 | −1.191 | −0.889 |
| GME | **cvar** | **+0.145** | **+0.186** | −1.296 | −1.000 |
| DOGE | none (baseline) | +3.953 | +5.815 | −0.132 | +1.429 |
| DOGE | cvar | +3.365 | +5.845 | −0.188 | +1.067 |
| PEPE | none (baseline) | +0.373 | +0.631 | −0.581 | −0.075 |
| PEPE | **cvar** | **−1.351** | **−1.963** | −0.755 | −0.629 |
| SHIB | none (baseline) | +0.885 | +0.803 | −0.558 | +0.093 |
| SHIB | cvar | +1.048 | +1.271 | −0.896 | −0.494 |
| **Basket avg** | none | −0.127 | +0.271 | −0.675 | −0.070 |
| **Basket avg** | cvar | **+0.456** | **+0.902** | **−0.859** | **−0.431** |

Bold = the more favorable Sharpe/Sortino side of each pair, to make the
pattern below easier to scan.

## What actually happened — an honest read, not the clean story hoped for

**The good news, as predicted:** AMC and GME — the two catastrophic
failures this upgrade specifically targeted — both improve substantially
in risk-adjusted terms. GME's Sharpe flips from deeply negative (−3.82) to
mildly positive (+0.15); AMC's improves by more than half (−2.03 → −0.93).
Basket-average Sharpe goes negative-to-positive (−0.127 → +0.456) and
Sortino nearly quadruples (+0.271 → +0.902). **DOGE, the one asset that was
already good, is not destroyed** — Sharpe gives back some ground (3.95 →
3.37) but stays far ahead of momentum (1.25) and Sortino is essentially
unchanged (5.815 → 5.845). This is exactly the gating condition Upgrade 1's
own dev prompt asked to check before extending further, and it passes.

**The finding the dev prompt did not anticipate: cumulative return and max
drawdown get WORSE under CVaR, for every single asset, no exceptions** —
basket-average cumulative return roughly sextuples in magnitude on the loss
side (−0.070 → −0.431) and max drawdown deepens (−0.675 → −0.859). **PEPE
is a clean counter-example to the whole hypothesis**: it was the basket's
second-best Sharpe under the original objective (+0.373) and becomes the
worst under CVaR (−1.351) — CVaR made a working policy meaningfully worse,
not just "didn't help."

**Mechanistic explanation, grounded in the actual action distributions**
(via `src/eval/reward_hacking_check.py` on every checkpoint, none/cvar
compared — see `reward_hacking_findings.md` for the original baseline
numbers, all 5 assets re-run here for the cvar side too):

| Asset | `frac_near_max_short` (none → cvar) | `std_action` (none → cvar) |
|---|---|---|
| AMC | 0.371 → **0.771** | 0.709 → 0.441 |
| SHIB | 0.262 → **0.583** | 0.583 → 0.581 |
| GME | 0.314 → 0.300 | 0.657 → 0.601 |
| PEPE | 0.500 → 0.462 | 0.586 → 0.593 |
| DOGE | (long-biased; `frac_near_max_long` 0.223 → 0.311) | 0.658 → 0.552 |

AMC and SHIB — the two clearest Sharpe/Sortino winners under CVaR — both
show a large jump toward `frac_near_max_short` (37%→77% for AMC, 26%→58%
for SHIB) *combined with* flat-to-lower action variance: the CVaR policy
isn't betting more cautiously, it's betting **more consistently** on the
same side. A steadier position (fewer direction flips) produces a smoother
day-to-day return series almost by construction, which mechanically
improves Sharpe/Sortino (both are mean-over-std-of-returns ratios) even
when the underlying directional call is unchanged or still wrong (SHIB's
`mean_action` actually got MORE negative, −0.37 → −0.66, while price action
in that window still cost it money — cumulative return went from +9% to
−49%). **This is the core, non-obvious finding of this ablation: the
implemented CVaR mechanism is rewarding consistency of exposure, not
correctness of the underlying directional call** — a real and useful
effect (a smoother equity curve is genuinely less likely to blow up an
account), but not the "learns to hedge / size down under uncertainty"
behavior the upgrade doc's hypothesis (section on Upgrade 1) described
wanting to see. That specific behavior — sizing DOWN during high-
uncertainty periods — was not observed; if anything, the CVaR agents got
more decisively committed, not more cautious.

A plausible reason: `predict_values()` returning `CVaR_0.2` (a pessimistic,
below-mean value estimate) systematically shifts GAE's advantage estimates
upward relative to the true mean-value baseline (since advantage = return −
V, and V is now smaller). This inflates the effective magnitude of
advantages across the board, which could push the policy toward larger,
more decisive updates in whichever direction the (now-noisier-in-scale)
advantage signal points — a bias-of-value-baseline effect, not a
genuine tail-risk-avoidance effect. This is a testable hypothesis for
follow-up work (e.g. comparing CVaR advantage magnitudes against
mean-value advantage magnitudes directly) but was not verified further in
this pass — flagged here as the leading explanation, not a confirmed one.

## Scope note — single window, not walk-forward

Per this ablation's time budget, all 10 runs above (5 assets × 2
objectives) are Phase-4a-style single-window comparisons, matching the
protocol the ORIGINAL "none" checkpoints were evaluated with (100k
timesteps, one train/validate split, seed 42) — not Phase 5's walk-forward
harness. `docs/CURRENT_STATUS.md` already documents that single-window and
walk-forward numbers disagree substantially even for the original PPO
runs (DOGE: 3.95 single-window vs 1.71 walk-forward average) — the same
caveat applies here with at least as much force, arguably more, since
CVaR's "smoother ratio, worse terminal outcome" pattern found above is
exactly the kind of thing a single short window could either overstate or
understate. **Walk-forward validation of the CVaR objective (retraining
per roll, the same way `src/eval/walk_forward.py` already does for the
"none" objective) is the single most important follow-up to this
document** before drawing any stronger conclusion than "CVaR trades
Sharpe/Sortino for cumulative-return/drawdown in a way that isn't uniformly
good or bad" — a single 70-103 day window is not enough data to know
whether the pattern found here is real signal or this window's noise.

## Conclusion and recommendation

**Do not treat CVaR-PPO as a strict upgrade over the original objective.**
It is a genuine, working, differently-shaped tradeoff: better Sharpe/
Sortino on 4 of 5 assets (all but PEPE), worse cumulative return and max
drawdown on all 5. Whether that tradeoff is "better" depends entirely on
what the final report chooses to optimize for and how that choice is
justified — reporting BOTH objectives side by side (as this document does)
rather than picking whichever looks better per-asset is the honest
framing, consistent with this project's existing standard (see
`docs/final_report.md`'s treatment of single-window vs walk-forward
disagreement for the same reasoning applied to a different axis).
`models_cvar/` is kept alongside `models/` specifically so both remain
available for whatever Upgrade 4 (multi-seed reporting) or a walk-forward
rerun decides to do with them, rather than only the "none" checkpoints
surviving as the sole record.
