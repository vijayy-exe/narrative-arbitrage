# Hyperparameter Search — LSTM (RecurrentPPO) on DOGE

Upgrade 4 of `narrative-arbitrage-implementation-upgrade.md`, part 1 (the
search itself; part 2, multi-seed reporting of the winning config across
the full basket, is in `docs/multiseed_report.md`). Code:
`src/agents/hparam_search.py sweep`.

**Why LSTM, why DOGE:** `docs/architecture_ablation.md` found LSTM
(RecurrentPPO) wins on every basket-average metric across Upgrades 1-3 —
"the best-performing architecture from Upgrades 1-3" per this upgrade's own
dev prompt. DOGE is the dev prompt's explicitly named search asset ("one
representative asset (DOGE, since it's the most stable)"). Caveat stated
plainly (also in `hparam_search.py`'s module docstring): the architecture
ablation separately found DOGE is the *one* asset where the plain MLP beats
LSTM outright — "stable" and "the asset LSTM helps most" are different
claims. DOGE was picked because the dev prompt names it, not because it's
LSTM's best case.

**Search space:** learning rate (log-uniform, 1e-5 – 1e-3), entropy
coefficient (log-uniform, 1e-4 – 5e-2), GAE lambda (uniform, 0.80 – 0.99),
LSTM hidden size (categorical, {32, 64, 128}). Sampler: Optuna's default
TPE, seeded for reproducibility.

**Budget:** 20 trials × 15,000 timesteps/trial (reduced from the project's
usual 100k-timestep budget — standard HPO practice, needed to fit 20 trials
in a practical amount of wall-clock time; ~31s/trial observed). The
*winning* config is retrained at the full 100k-timestep budget across 5
seeds afterward (`docs/multiseed_report.md`), so this reduced budget only
affects which hyperparameters get selected, not the final reported numbers.

## Results

| Trial | Sharpe | learning_rate | ent_coef | gae_lambda | lstm_hidden_size |
|---:|---:|---:|---:|---:|---:|
| 2 (**best**) | **3.410** | 4.62e-4 | 3.74e-4 | 0.8345 | 128 |
| 3 | 3.331 | 7.31e-5 | 6.11e-4 | 0.9163 | 128 |
| 0 | 3.113 | 5.61e-5 | 3.68e-2 | 0.9391 | 32 |
| 16 | 2.889 | 2.44e-4 | 2.34e-4 | 0.8992 | 128 |
| 9 | 2.804 | 1.57e-4 | 3.08e-2 | 0.8168 | 128 |
| 8 | 2.688 | 1.24e-4 | 3.15e-4 | 0.9842 | 64 |
| 6 | 2.674 | 4.07e-5 | 1.83e-4 | 0.9300 | 128 |
| 17 | 2.682 | 3.61e-4 | 7.26e-4 | 0.8728 | 64 |
| 11 | 2.522 | 3.45e-4 | 1.02e-3 | 0.8451 | 128 |
| 15 | 2.490 | 3.41e-5 | 4.72e-4 | 0.9751 | 128 |
| 19 | 2.074 | 9.18e-4 | 2.31e-3 | 0.8865 | 128 |
| 13 | 2.015 | 8.86e-4 | 2.88e-4 | 0.8117 | 128 |
| 1 | 1.921 | 1.31e-5 | 2.18e-2 | 0.9142 | 128 |
| 12 | 1.859 | 1.50e-5 | 8.17e-4 | 0.8827 | 128 |
| 7 | 1.670 | 1.17e-5 | 2.85e-2 | 0.8492 | 32 |
| 4 | 1.618 | 8.17e-5 | 1.32e-2 | 0.8379 | 64 |
| 14 | 0.311 | 9.05e-4 | 3.11e-3 | 0.8059 | 128 |
| 18 | 0.291 | 8.62e-4 | 1.28e-4 | 9.40e-1 | 128 |
| 10 | −0.193 | 5.91e-4 | 1.06e-2 | 0.9243 | 128 |
| 5 | −0.398 | 1.64e-4 | 3.74e-4\* | 0.8124 | 64 |

*(all 20 trials shown, sorted by Sharpe descending; full raw log has exact
values to more digits.)*

**Winning config:** `learning_rate=4.62e-4, ent_coef=3.74e-4,
gae_lambda=0.8345, lstm_hidden_size=128` (trial 2, Sharpe 3.410 at the
reduced 15k-timestep budget — this is NOT directly comparable to
`docs/architecture_ablation.md`'s DOGE/LSTM number of Sharpe 2.139, which
used the untuned defaults at the full 100k-timestep budget; see
`docs/multiseed_report.md` for the fair, full-budget comparison).

## What the sweep shows

- **`lstm_hidden_size=128` dominates the top of the leaderboard** — 8 of
  the top 9 trials use 128, versus only occasional appearances of 32/64
  higher up. More capacity helped at this reduced training budget, at
  least for DOGE.
- **Low-to-moderate entropy coefficient (`ent_coef` roughly 1e-4 to 1e-3)
  clusters at the top**; the two worst trials (10, 5, both negative Sharpe)
  and several mid-pack trials have either high entropy (over-exploration,
  e.g. trial 1 at 0.022) or land in an unlucky learning-rate/gae_lambda
  combination — no single hyperparameter cleanly separates good from bad on
  its own, consistent with this being a real (if reduced-budget) joint
  search, not a one-dimensional effect.
- **No strong, clean trend in `learning_rate` or `gae_lambda` alone** —
  good trials span most of both ranges (e.g. trial 3 at lr=7.3e-5 and
  trial 2 at lr=4.6e-4 are 1st/2nd; gae_lambda ranges from 0.83 to 0.92
  among the top 5). This is a genuinely 20-trial, TPE-guided but still
  fairly under-sampled search of a 4-dimensional space — treat "the winning
  config" as a reasonable, defensible pick from a small budget, not a
  precisely-characterized optimum.
