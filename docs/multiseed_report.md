# Multi-Seed Report — Tuned LSTM, All 5 Assets

> **⚠ UPDATE (2026-09-20):** the LSTM architecture this hyperparameter
> search was built on does not beat the original plain MLP under
> walk-forward re-validation (basket-average Sharpe 0.451 vs. 0.712 — see
> `docs/walk_forward_upgrade_validation.md`), and the sweep asset (DOGE)
> is specifically the one asset where walk-forward shows LSTM performing
> worse (0.701) than the original MLP (1.712). The seed-variance finding
> below is still real and still applies to every single-seed number in
> this project generally — that methodological point doesn't depend on
> which architecture was tuned — but the specific config tuned here is no
> longer the recommended one for DOGE.

Upgrade 4 of `narrative-arbitrage-implementation-upgrade.md`, part 2 (part
1, the hyperparameter sweep itself, is in `docs/hparam_search_results.md`).
Code: `src/agents/hparam_search.py multiseed`.

**Config used** (the sweep's DOGE-tuned winner, applied UNCHANGED to all 5
assets — no per-asset re-tuning, at the full 100k-timestep budget, not the
sweep's reduced 15k budget): `learning_rate=4.62e-4, ent_coef=3.74e-4,
gae_lambda=0.8345, lstm_hidden_size=128`. 5 seeds (100-104) per asset, 25
runs total.

## Results — per asset, all 5 seeds

| Asset | Seed 100 | Seed 101 | Seed 102 | Seed 103 | Seed 104 | **Mean ± std** |
|---|---:|---:|---:|---:|---:|---:|
| DOGE | 3.510 | **−1.746** | 2.236 | 2.238 | 2.360 | **1.719 ± 1.798** |
| AMC | 2.096 | −0.884 | **3.372** | −0.819 | 0.956 | **0.944 ± 1.654** |
| GME | 1.207 | 0.841 | 1.012 | **−1.302** | 1.656 | **0.683 ± 1.029** |
| PEPE | 0.839 | 1.281 | 1.317 | **2.488** | 0.297 | **1.245 ± 0.723** |
| SHIB | 1.613 | 0.267 | 1.772 | 1.366 | **−0.455** | **0.913 ± 0.863** |
| **Basket avg** | | | | | | **1.101 ± n/a** |

(Sharpe Ratio shown; bold marks each row's min/max seed. Basket avg is the
mean of the 5 per-asset means — 1.101, essentially the same as
`docs/architecture_ablation.md`'s single-seed-42, untuned-hyperparameter
LSTM basket average of 1.161. **The tuned config's average performance is
not meaningfully different from the untuned single-seed number it was
supposed to improve on — the entire value of tuning shows up as reduced
variance for some assets (PEPE: std 0.72, the tightest of the five) and
unchanged-or-worse variance for others (AMC: std 1.65, the widest), not as
a clear across-the-board mean improvement.**)

## Every asset has at least one seed that flips sign or is a clear outlier

**Every single asset in the basket has at least one seed whose Sharpe is
either negative or a dramatic outlier from its own other 4 seeds** — DOGE
(seed 101: −1.75), AMC (seeds 101/103: −0.88/−0.82, though seed 102 is a
huge +3.37 outlier the other direction), GME (seed 103: −1.30), PEPE (seed
103 is a positive outlier at +2.49, not negative, but still a 2x deviation
from the other four), SHIB (seed 104: −0.46). **This is not a DOGE-specific
fluke — it is a basket-wide property of this architecture/config
combination.** Any report that cites a single seed's Sharpe for this LSTM
config, for any of these 5 assets, has a roughly 1-in-5 chance (by this
sample) of citing the unlucky (or luckily flattering) draw rather than a
representative one.

## The headline finding: this project's seed=42 point estimates were never as solid as they looked

**The standard deviation across seeds (1.798) is nearly as large as the
mean Sharpe itself (1.719).** Four of five seeds land in a tight, good
cluster (2.24 – 3.51); the fifth (seed 101) is not just worse, it's
**negative** (−1.746) — with the *identical* hyperparameters, data, and
architecture, changing only the random seed. This is the exact failure mode
Upgrade 4's own dev prompt existed to catch ("Single seed 42... is not
defensible as 'the result'"), and it turns out to be real, not a
theoretical concern: **every other single-seed number in this entire
project — every table in `docs/backtest_report.md`, `docs/
architecture_ablation.md`, `docs/risk_sensitivity_ablation.md`, all of
them — carries this same unquantified seed risk.** DOGE's own
architecture-ablation LSTM result (Sharpe 2.139, seed 42) sits comfortably
inside this run's good cluster, so that particular number wasn't
seed-unlucky — but there was no way to know that without this multi-seed
run, and there is no guarantee the other 14 single-seed numbers in that
same table are equally lucky.

**What this does and doesn't undermine:**
- It does **not** overturn `docs/architecture_ablation.md`'s LSTM-wins-on-
  average finding — the basket-average mean across seeds (1.101) is close
  to that single-seed-42 basket average (1.161), so the headline
  conclusion happens to survive multi-seeding at the basket-average level.
- It **does** mean any PER-ASSET, PER-SEED number quoted from the
  architecture or CVaR ablations should be read as "true for that one
  seed," not "true for the architecture/objective in general" — every
  asset in this basket has at least one seed that flips sign or is a
  dramatic outlier (see above), so a different single seed could easily
  have produced a different per-asset ranking, even if the basket average
  is more stable than any individual asset's number.
- **This retrofit is now done for the LSTM/tuned-hyperparameter config,
  across all 5 assets.** What remains NOT multi-seeded is the CVaR
  objective (Upgrade 1) and the Transformer/untuned-MLP architectures
  (Upgrades 2-3) — each still rests on a single seed=42 run per asset. See
  `docs/final_report.md` section 8 for the current state of that broader
  retrofit.

## Comparison to the single-seed numbers this config was tuned against

`docs/architecture_ablation.md`'s LSTM entries (untuned defaults, seed 42,
full 100k budget) vs. this run's tuned, multi-seed mean:

| Asset | Untuned, seed=42 (Sharpe) | Tuned, mean±std across 5 seeds |
|---|---:|---:|
| DOGE | 2.139 | 1.719 ± 1.798 |
| AMC | −1.048 | 0.944 ± 1.654 |
| GME | −0.035 | 0.683 ± 1.029 |
| PEPE | 1.825 | 1.245 ± 0.723 |
| SHIB | 2.923 | 0.913 ± 0.863 |

**Tuning (on DOGE, applied unchanged everywhere) helped AMC and GME's
single-seed-42 numbers (both improve) but actually made DOGE, PEPE, and
SHIB's single-seed-42 numbers look worse when compared to the tuned mean**
— though for all three, that single untuned seed=42 draw sits inside or
above the tuned config's normal seed-to-seed range, so "worse than the old
number" is at least partly an artifact of comparing one particular old
seed against a new distribution's mean, not proof the tuned config is
strictly worse for those three. The honest, complete answer to "did tuning
help" is asset-specific and seed-sensitive, not a clean yes — reporting
mean ± std (as the table above does) rather than cherry-picking either
side's best seed is the honest choice, consistent with this project's
established standard elsewhere (e.g. `docs/CURRENT_STATUS.md`'s treatment
of single-window vs. walk-forward disagreement).
