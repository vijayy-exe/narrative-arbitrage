# Uncertainty-Aware Position Sizing — Tier 2, Phase 5

`src/agents/uncertainty_critic.py`: a post-hoc wrapper, distinct from both
Upgrade 1's CVaR objective (reshapes what the critic's *distribution*
targets) and `src/agents/ensemble.py`'s seed ensemble (averages ACTIONS
across independently-trained actors). Here, one base policy's action is
scaled down by an independently-trained ensemble of 3 critics'
DISAGREEMENT on the current state's value — a standard epistemic-
uncertainty proxy, applied without retraining the base policy itself.
Walk-forward tested on AMC and GME (10 rolls each, 15k timesteps/roll,
seed 42, `--rl-variant uncertainty_aware`).

## Results — mixed, not a clean win

| Asset | Metric | Original PPO | Uncertainty-aware |
|---|---|---:|---:|
| AMC | Sharpe | 0.278 | **−0.650** (worse) |
| AMC | Max Drawdown | −0.231 | **−0.133** (better) |
| GME | Sharpe | −1.000 | **−0.481** (better) |
| GME | Max Drawdown | −0.130 | **−0.070** (better) |

**GME: a genuine, if modest, improvement on both axes** — better Sharpe
and better max drawdown than the unscaled baseline, consistent with the
hypothesis that scaling down during high-disagreement states helps. Still
far short of the LSTM ensemble's 2.139 (`docs/
walk_forward_upgrade_validation.md`) — this wrapper is not a substitute
for the architecture change that actually worked well for GME.

**AMC: drawdown improves, Sharpe gets meaningfully worse.** The wrapper
reduced AMC's max drawdown (−0.231 → −0.133, a real risk reduction) but at
a real cost to risk-adjusted return — AMC's original PPO policy, imperfect
as it was, apparently had enough of its "reckless" behavior actually
contributing to (small) positive Sharpe that damping it hurt more than it
helped. This is the same lesson `docs/risk_sensitivity_ablation.md` and
`docs/joint_portfolio_validation.md` both already found with a different
mechanism: **risk-reduction techniques on this project's data reliably
trade Sharpe for drawdown, not a free lunch on both** — this wrapper is a
third independent confirmation of that pattern, using yet another
mechanism (epistemic uncertainty, not CVaR reshaping or joint-policy
smoothing).

## Honest scope note

Only 3 critic ensemble members, `k=2.0` scaling coefficient untuned (no
sweep attempted — unlike Phase 3's Pareto frontier, which deliberately
swept its analogous hyperparameter), single seed, 2 assets. The mixed
result above is real but should be read as "this specific configuration,"
not "epistemic-uncertainty sizing definitively does/doesn't work" —
sweeping `k` the way Phase 3 swept `cvar_alpha` is the natural next step
if this line is pursued further, not attempted here given time budget.
