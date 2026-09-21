# Pareto Frontier -- CVaR's Risk/Return Tradeoff, Made Visible

`CVaRPPO`'s own `cvar_alpha` parameter swept across [0.9, 0.7, 0.5, 0.3, 0.1], walk-forward evaluated (15000 timesteps/roll, seed=42) on ['GME', 'AMC']. See src/eval/pareto_frontier.py's module docstring for why this reuses existing, already-tested CVaRPPO infrastructure rather than a new algorithm.

## GME

|   cvar_alpha |   Sharpe Ratio |   Sortino Ratio |   Max Drawdown |   Cumulative Return |
|-------------:|---------------:|----------------:|---------------:|--------------------:|
|          0.9 |        -0.9944 |         -0.463  |        -0.0959 |             -0.0168 |
|          0.7 |        -0.9839 |         -0.1281 |        -0.0955 |             -0.0147 |
|          0.5 |        -0.8869 |          0.0177 |        -0.0908 |             -0.0109 |
|          0.3 |        -0.7037 |          0.354  |        -0.0824 |             -0.0017 |
|          0.1 |        -0.6216 |          1.0244 |        -0.0638 |              0.0065 |

Reference points (already-completed walk-forward runs, not recomputed here):

- **Original PPO**: Sharpe -0.9995, Max Drawdown -0.1296
- **CVaR (alpha=0.2, original)**: Sharpe -0.5626, Max Drawdown -0.0730

## AMC

|   cvar_alpha |   Sharpe Ratio |   Sortino Ratio |   Max Drawdown |   Cumulative Return |
|-------------:|---------------:|----------------:|---------------:|--------------------:|
|          0.9 |        -0.0084 |          2.8872 |        -0.1573 |             -0.0324 |
|          0.7 |         0.0351 |          3.2059 |        -0.1548 |             -0.0376 |
|          0.5 |         0.1828 |          4.3573 |        -0.1566 |             -0.0409 |
|          0.3 |         0.1182 |          5.5434 |        -0.1553 |             -0.0475 |
|          0.1 |        -1.1006 |          2.1015 |        -0.1508 |             -0.0734 |

Reference points (already-completed walk-forward runs, not recomputed here):

- **Original PPO**: Sharpe 0.2780, Max Drawdown -0.2314
- **CVaR (alpha=0.2, original)**: Sharpe -0.4490, Max Drawdown -0.1546

## What the frontier actually shows — asset-dependent, not one clean tradeoff

**GME: not really a tradeoff at all — lower alpha is a win-win.** Sharpe
improves monotonically as alpha decreases (−0.994 → −0.622) AND max
drawdown improves monotonically at the same time (−0.096 → −0.064). More
aggressive tail-focus makes GME's CVaR policy better on both axes
simultaneously in this sweep, not a tradeoff a Pareto frontier would
normally show as a curve — it's closer to a straight dominance ordering.
The best CVaR point here (alpha=0.1, Sharpe −0.622) actually beats
original PPO's own GME number (−0.9995) — so aggressive CVaR tuning does
help GME relative to plain PPO — though it's still far short of the LSTM
ensemble's 2.139 (`docs/walk_forward_upgrade_validation.md`). CVaR tuning
helps GME, but nowhere near as much as the architecture change did.

**AMC: a genuine Pareto tradeoff, peaking well short of original PPO.**
Sharpe rises from alpha=0.9 (−0.008) to a peak around alpha=0.5 (0.183),
then collapses at alpha=0.1 (−1.101) — a real inverted-U, not a monotonic
trend. But **no point on this curve beats AMC's original PPO Sharpe
(0.278)** — CVaR tuning, at any alpha tried, cannot recover what AMC's
attention-only policy already had without a risk-sensitive objective at
all. This directly answers the question this analysis set out to ask:
for AMC, CVaR is not a legitimate point on a favorable tradeoff curve
the original Sharpe-only comparison hid — it's a strictly worse choice
across the whole alpha range swept, exactly as the blunter single-number
comparison originally suggested.

**Reproducibility caveat, stated plainly:** the original alpha=0.2 CVaR
reference point (GME Sharpe −0.563) is better than BOTH of this sweep's
neighboring points (alpha=0.1: −0.622, alpha=0.3: −0.704) — breaking
strict monotonicity. Both runs used `seed=42`, but `seed` only controls
`PPO`'s own RNG, not every source of nondeterminism across separate
process launches (PyTorch's global RNG state, action-sampling order, etc.)
-- so exact point-for-point reproducibility across two different script
invocations, even at "the same" configuration, isn't guaranteed here.
This is a smaller-scale echo of `docs/multiseed_report.md`'s seed-variance
finding, not a new problem: individual points on this curve carry some of
the same run-to-run noise as every other single-run number in this
project. The overall SHAPE of each asset's curve (GME monotonic, AMC
inverted-U) is more trustworthy than any single point's exact value.

