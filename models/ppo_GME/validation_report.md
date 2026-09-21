# PPO Training Report -- GME

Train window: start -> 2024-02-20

Validation window: 2024-02-21 -> 2024-05-31

Total timesteps: 100000, seed: 42

## PPO vs momentum baseline (same validation window)

|                        |   Cumulative Return |   Sharpe Ratio |   Sortino Ratio |   Max Drawdown |
|:-----------------------|--------------------:|---------------:|----------------:|---------------:|
| PPO (this run)         |           -0.889313 |       -3.81616 |        -3.43716 |      -1.19081  |
| Momentum (same window) |            0.174575 |        1.16548 |         2.37693 |      -0.624205 |

Mean |action|: 0.5786, std(action): 0.6571 -- see src/eval/reward_hacking_check.py for a full action-collapse check.
