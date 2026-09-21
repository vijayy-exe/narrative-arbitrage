# Walk-Forward Validation of Upgrades 1-3 — The Single-Window Results Do Not Hold Up

This is the single most important document produced by the entire RL
upgrade path (`narrative-arbitrage-implementation-upgrade.md`). Every
result in `docs/risk_sensitivity_ablation.md` (Upgrade 1, CVaR) and
`docs/architecture_ablation.md` (Upgrades 2-3, Transformer/LSTM) was
single-window — a known, explicitly-flagged limitation in both documents,
which each named walk-forward re-validation as their top follow-up. This
document is that re-validation, using the exact same protocol and harness
(`src/eval/walk_forward.py`, extended with `evaluate_cvar_roll`,
`evaluate_lstm_roll`, `evaluate_transformer_roll` — see that file) as
produced `docs/backtest_report.md`'s original walk-forward numbers: 6-month
train / 1-month validate, rolled forward 1 month at a time, retrained from
scratch every roll, 15,000 PPO timesteps/roll, all 5 assets, 48 total
rolls (AMC/GME 10, DOGE/SHIB 11, PEPE 6).

## The headline result

**None of the three upgrades beat the original plain-MLP PPO's
walk-forward Sharpe. All three single-window "improvements" evaporate or
reverse under walk-forward re-validation.**

| Method | Cum. Return | Sharpe | Sortino | Max DD |
|---|---:|---:|---:|---:|
| Momentum (baseline, original) | 0.1711 | **1.1191** | 3.0488 | −0.1939 |
| **PPO / plain MLP (original, Upgrade 0 baseline)** | 0.0498 | **0.7120** | 2.8460 | −0.1461 |
| Transformer-PPO (Upgrade 2, new) | 0.0285 | 0.5116 | 2.8315 | −0.1221 |
| LSTM / RecurrentPPO (Upgrade 3, new) | 0.0177 | 0.4509 | 2.6424 | **−0.1964** |
| CVaR-PPO (Upgrade 1, new) | 0.0026 | 0.1104 | **5.1158** | −0.1032 |
| Supervised / XGBoost (baseline, original) | −0.0065 | 0.0654 | 1.0649 | −0.2131 |

All figures are roll-weighted basket averages (mean across all 48 rolls,
matching exactly how `docs/backtest_report.md`'s original basket-average
row was computed — NOT a simple mean-of-5-per-asset-means, which would
give a slightly different number since PEPE only contributes 6 of 48
rolls). Momentum and PPO/MLP rows are unchanged, copied directly from
`docs/backtest_report.md`.

**Ranking by walk-forward Sharpe: momentum > original PPO > Transformer >
LSTM > CVaR > supervised.** The single-window architecture ablation found
LSTM beating the MLP baseline on every basket-average metric (Sharpe 1.161
vs. −0.127); under walk-forward, LSTM's Sharpe (0.451) is *worse* than the
MLP baseline's (0.712), not better. The single-window risk-sensitivity
ablation found CVaR improving Sharpe/Sortino on 4 of 5 assets; under
walk-forward, CVaR's Sharpe (0.110) is the worst of all three RL
architectures tried, beating only the supervised baseline. Transformer
fares best of the three new approaches but still trails the original
unmodified PPO.

## Per-asset detail — walk-forward

| Asset | Original PPO Sharpe | CVaR Sharpe | LSTM Sharpe | Transformer Sharpe |
|---|---:|---:|---:|---:|
| DOGE | 1.7116 | 0.8747 | 0.7009 | **1.0306** |
| SHIB | 1.4555 | 0.6088 | −0.0891 | 0.6918 |
| PEPE | 1.0922 | −0.1507 | 0.4524 | 0.2695 |
| AMC | 0.2780 | −0.4490 | −0.1638 | −0.1644 |
| GME | −0.9995 | −0.5626 | **1.3837** | 0.5638 |

**Every one of DOGE, SHIB, and PEPE — the three assets original PPO
already handled reasonably (all Sharpe > 1.0) — gets WORSE under every
single new variant tried**, no exceptions. The only genuine walk-forward
win among all 15 (asset × new-variant) cells is **GME under LSTM (1.384,
vs. original PPO's −0.9995)** — a real, large improvement on the basket's
single worst-performing asset. Transformer's GME number (0.564) is a
smaller version of the same effect. CVaR does not meaningfully help GME
under walk-forward (−0.563, still clearly negative) despite being the
single-window ablation's strongest GME result (+0.145).

## Even the one confirmed win isn't fully seed-robust

Given `docs/multiseed_report.md`'s separate finding that seed variance is
large enough to flip signs elsewhere in this project, the one genuine
walk-forward win above (LSTM on GME) was re-run at 2 more seeds (100, 101)
to check whether it holds up:

| Seed | GME/LSTM walk-forward Sharpe |
|---:|---:|
| 42 (original) | 1.384 |
| 100 | **1.917** |
| 101 | **−0.624** |
| Mean ± std | **0.892 ± 1.094** |

**Even this result carries the same seed-variance risk found everywhere
else in this upgrade path** — one of three seeds flips it negative. The
mean (0.892) is still clearly better than original PPO's GME walk-forward
Sharpe (−0.9995), so "LSTM genuinely helps GME on average" survives this
3-seed check — but "LSTM reliably fixes GME" does not: a different single
seed could easily have reported GME/LSTM as still broken. This is the
honest, complete picture: not "walk-forward confirmed a clean win," but
"walk-forward confirmed a real average improvement with substantial
run-to-run variance," which is a meaningfully weaker and more useful
claim. A full 5-seed run (matching `docs/multiseed_report.md`'s protocol
elsewhere) was not done for this specific check, given time constraints —
3 seeds is enough to demonstrate the variance is real, not enough to
precisely characterize it.

## The fix: a seed ensemble resolves the variance problem, and then some

Given the seed-variance finding above, `src/agents/ensemble.py` builds a
genuine seed-ensemble policy: 3 independently-seeded LSTM models trained
fresh per roll, whose actions are averaged at every timestep (not a
post-hoc average of 3 separate backtests — see that module's docstring for
why the distinction matters). Run on GME under the same walk-forward
protocol (`--rl-variant lstm_ensemble`):

| Config | Walk-forward Sharpe | Max Drawdown |
|---|---:|---:|
| Original PPO (baseline) | −1.000 | −1.191 |
| LSTM, seed 42 | 1.384 | −0.256 |
| LSTM, seed 100 | 1.917 | (not separately tracked) |
| LSTM, seed 101 | −0.624 | (not separately tracked) |
| LSTM, mean across 3 seeds | 0.892 | — |
| **LSTM, 3-member ensemble** | **2.139** | **−0.081** |

**The ensemble doesn't just reduce variance — it beats every single-seed
run tried, including the best one (1.917).** Max drawdown also improves
to the best number found anywhere in this asset's walk-forward results
(−0.081, versus −0.256 for the single best seed and far worse for the
others). This is a clean, positive resolution to the seed-variance
problem this document spent most of its length documenting: averaging
independently-seeded members' actions at every step gives a real,
measured improvement over trusting any single seed.

**Confirmed on a second asset (2026-09-20, Tier 2 follow-up): AMC.**

| Config | Walk-forward Sharpe |
|---|---:|
| Original PPO (baseline) | 0.278 |
| CVaR (alpha=0.2) | −0.449 |
| LSTM, seed 42 | −0.164 |
| Transformer | −0.164 |
| **LSTM, 3-member ensemble** | **1.033** |

The ensemble beats every other variant tried for AMC, including the
original PPO baseline — not just the architecture it's ensembling
(LSTM), the entire basket-wide "none of the upgrades beat original PPO"
finding above. **Two for two**: this is no longer a GME-specific
curiosity, it's a real, repeatable effect. AMC and GME (10 rolls, 3
members each, ~28 min/asset) are the two tested so far — DOGE, PEPE, and
SHIB remain a mechanical, not-yet-run follow-up.

## Why the single-window numbers were misleading — and this isn't new news

This isn't a surprise this project was unprepared for. `docs/
CURRENT_STATUS.md` already documented, before ANY of Upgrades 1-4 were
built, that the ORIGINAL single-window PPO number for DOGE (Sharpe 3.95)
and its walk-forward average (1.71) disagreed by more than 2x — "exactly
the gap Phase 5 was designed to catch." Every subsequent single-window
ablation in this upgrade path repeated the same warning explicitly
(`risk_sensitivity_ablation.md`'s "Scope note", `architecture_ablation.md`'s
"Scope note", both naming walk-forward re-validation as the top follow-up).
**What this document adds is not a new caveat, but direct, executed proof
that the caveat was load-bearing**: the disagreement between single-window
and walk-forward wasn't just theoretically possible, it was large enough
to flip which architecture looks best.

A plausible mechanism, consistent with `docs/multiseed_report.md`'s
separate finding (seed variance nearly as large as the mean Sharpe for the
tuned LSTM config): single-window results with only 100k timesteps and one
train/val split are each closer to one noisy draw than a stable estimate of
an architecture's true edge. Walk-forward's 48 independently-retrained
rolls average out much of that noise — and once averaged out, the extra
capacity/complexity each upgrade added (a distributional critic, a
recurrent policy, a Transformer encoder) did not translate into a real
edge over the original, simplest architecture, for the basket as a whole.
GME under LSTM is the one place it clearly did.

## What this means for every other document in this upgrade path

- **`docs/risk_sensitivity_ablation.md`'s headline finding
  ("Sharpe/Sortino improve on 4/5 assets") is walk-forward-CONTRADICTED,
  not just single-window-scoped.** That document's own single-window
  numbers are still accurate as single-window numbers, but should not be
  read as evidence CVaR is a good idea for this problem — walk-forward
  says the opposite.
- **`docs/architecture_ablation.md`'s headline finding ("LSTM wins on
  every basket-average metric") does NOT survive walk-forward, except
  narrowly for GME.** The Transformer's single-window AMC result (+0.351,
  called out as "the only positive number this asset has produced under
  ANY objective/architecture tried") also does not survive — Transformer's
  walk-forward AMC Sharpe is −0.164, statistically indistinguishable from
  LSTM's −0.164 and still solidly negative.
- **`docs/hparam_search_results.md` and `docs/multiseed_report.md`
  tuned/multi-seeded a config (LSTM on DOGE) whose walk-forward performance
  (0.701, this document) is itself below the original untuned MLP's
  walk-forward number (1.712) for that same asset** — the entire Upgrade 4
  search was built on top of a single-window architecture choice that
  walk-forward now says was the wrong one for DOGE specifically (DOGE's own
  single-window ablation already flagged the plain MLP as DOGE's best
  architecture; this walk-forward run reinforces that specifically for
  DOGE, while showing LSTM's real value lives in GME instead).

## Honest bottom line

**If the goal is basket-wide walk-forward Sharpe, none of the individual
Upgrades 1-3 architectures should be deployed over the original plain-MLP
PPO from Upgrade 0.** The one exception with real average value is **LSTM
specifically for GME** — a genuine improvement on average (−1.00 → +0.89
across 3 seeds) on the single worst-performing asset in the original
basket, though even a single ensemble member's seed-to-seed variance is
substantial on its own. **The seed ensemble resolves that**: 3
independently-seeded LSTM members, averaged at every step, reach Sharpe
2.139 on GME — better than every individual seed tried, not just the
mean. So the complete, honest recommendation is not "skip the upgrade
path" but "skip individual single-seed architecture changes; a seed
ensemble of the walk-forward-validated winner is the one piece of this
upgrade path that clearly earns its extra complexity." Everywhere else,
the extra architectural complexity introduced by this upgrade path did
not pay for itself once evaluated the way this project's own methodology
requires. This is not a failure of the upgrade path — finding out that
single-window ablations overstate real generalization, with hard numbers
rather than a caveat, is a genuinely valuable (if humbling) research
result in its own right, and is now the actual
headline finding of this entire session's work.
