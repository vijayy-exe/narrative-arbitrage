# Reward-Hacking / Policy-Collapse Audit — AMC & GME

Run per `narrative-arbitrage-implementation-upgrade.md` Upgrade 0, step 2,
in response to `docs/CURRENT_STATUS.md`'s flag that AMC (single-window
Sharpe −2.03) and GME (−3.82) are the basket's two worst performers and
warrant scrutiny before any further changes to their training. Command:
`python -m src.eval.reward_hacking_check --asset {AMC,GME}`, using the
existing action-distribution collapse check
(`src/eval/reward_hacking_check.py`) against each asset's Phase-4a
single-window validation rollout. DOGE (the basket's best performer, single-
window Sharpe 3.95) was also run as a baseline for comparison.

## Result: not reward hacking

Neither AMC nor GME shows a collapsed/degenerate action distribution.

| Asset | mean action | std action | frac near-zero | frac near-max-long | frac near-max-short | Verdict |
|---|---:|---:|---:|---:|---:|---|
| DOGE (baseline) | 0.236 | 0.658 | 0.097 | 0.223 | 0.126 | OK — varies with input |
| AMC | −0.305 | 0.709 | 0.114 | 0.100 | 0.371 | OK — varies with input |
| GME | −0.236 | 0.657 | 0.143 | 0.086 | 0.314 | OK — varies with input |

All three action distributions clear the check's thresholds by a wide
margin (std ≈ 0.65–0.71, none of the three collapse fractions anywhere
near the 0.90 trigger). The policy is not sitting flat to game a
vol-normalized reward, and it is not pinned to always-max-long or
always-max-short. Whatever is producing AMC's and GME's bad Sharpe, it is
a **genuine (if bad) learned policy, not a degenerate one** — this check
gives no reason to distrust their attribution output on collapse grounds.
(It does not, on its own, clear them for the *other* reason
`docs/final_report.md` §5a's update note gives for holding off on
extending attribution to them — see that note.)

## What the bad Sharpe actually looks like: a directional miss, not noise

Cross-referencing each policy's mean action against the actual price
return over the same validation window (`data/raw/price/{asset}.csv`,
2024-02-21 → 2024-05-31 for AMC/GME, 2024-02-19 → 2024-06-01 for DOGE)
makes the failure mode legible:

| Asset | mean action (net bias) | price return over window | Read |
|---|---:|---:|---|
| DOGE | +0.236 (net long) | +79.3% | Correct direction, correct sizing intuition |
| GME | −0.236 (net short) | **+72.6%** | **Direct directional miss** — net short into a strong rally |
| AMC | −0.305 (net short) | −5.3% | Direction roughly right, but a strong net-short bias against a nearly-flat/mildly-down asset, plus `frac_near_max_short` = 0.37 (over a third of steps near max-short) — looks like overconfident sizing/overtrading rather than a sign error |

GME is the clean story: the policy learned a net-short bias and got
run over by a 72.6% rally — this alone accounts for most of the −3.82
Sharpe, no exotic explanation needed. AMC is murkier: the sign of the bias
roughly matches the (small) price move, so the loss is more likely coming
from position-sizing/timing/transaction-cost churn (over a third of
actions sit near max-short on an asset that barely moved) than from a
wrong-direction bet — this would need a per-step trade log to confirm
precisely, which is out of scope for this pass.

## Conclusion and how this affects the Upgrade path

- **Cleared for Upgrade 1 (risk-sensitive objective):** since both
  policies are genuinely learned, not hacked, `narrative-arbitrage-
  implementation-upgrade.md`'s Upgrade 1 (CVaR/distributional critic) is
  the right next lever — it directly targets exactly this kind of
  confident, wrong-directional or overtraded position sizing, and there is
  no hacked-reward confound to worry about layering it on top of.
- **GME's failure mode (directional miss) and AMC's (likely
  overconfident sizing/overtrading) may need different fixes** —
  worth keeping both hypotheses distinct when interpreting the Upgrade 1
  ablation results rather than expecting one mechanism to explain both.
- **Does not by itself clear AMC/GME for the attribution extension**
  discussed in `docs/final_report.md` §5a's 2026-09-20 update — that note's
  reasoning was "don't trust attribution on a possibly-hacked policy,"
  which this audit now rules out, but the attribution extension itself is
  still deferred pending a decision documented there, not automatically
  unblocked by this file alone.
