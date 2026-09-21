# PPO Training Report -- AMC

Train window: start -> 2024-02-20

Validation window: 2024-02-21 -> 2024-05-31

Total timesteps: 100000, seed: 42

## PPO vs momentum baseline (same validation window)

|                        |   Cumulative Return |   Sharpe Ratio |   Sortino Ratio |   Max Drawdown |
|:-----------------------|--------------------:|---------------:|----------------:|---------------:|
| PPO (this run)         |           -0.908085 |       -2.02983 |        -2.45636 |      -0.910838 |
| Momentum (same window) |            0.602971 |        1.62482 |         3.97313 |      -0.382482 |

Mean |action|: 0.6868, std(action): 0.7085 -- see src/eval/reward_hacking_check.py for a full action-collapse check.
