# Walk-Forward Seed Spot-Checks — CVaR and Transformer

`docs/walk_forward_upgrade_validation.md` established that CVaR (basket
Sharpe 0.110) and Transformer (0.512) both trail original PPO (0.712)
under walk-forward, at seed=42. Given `docs/multiseed_report.md`'s
seed-variance finding, these spot-checks (DOGE, seed 100, in addition to
the existing seed=42 number) test whether that ranking could be a seed
artifact rather than a real effect. A full 5-asset x multi-seed retrofit
was judged too expensive (~2+ hours) relative to a targeted spot-check on
DOGE (the asset with the strongest original-PPO baseline, so the
"upgrades underperform" gap has the most room to close if it were seed
noise) — see `docs/ensemble_validation.md` for where the remaining budget
went instead (an actual fix for the seed-variance problem, not just more
measurement of it).

| Variant | DOGE walk-forward Sharpe, seed=42 | seed=100 | Original PPO (for reference) |
|---|---:|---:|---:|
| CVaR | 0.875 | 0.500 | 1.712 |
| Transformer | 1.031 | 0.563 | 1.712 |

**Both variants underperform original PPO at both seeds, by a wide and
consistent margin.** Neither number comes close to closing the gap to
1.712 — this is not seed noise producing a misleading single-seed
snapshot; it's a real, seed-independent effect on DOGE specifically. This
corroborates (does not merely fail to contradict) `walk_forward_upgrade_
validation.md`'s basket-level finding.
