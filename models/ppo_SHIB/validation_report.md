# PPO Training Report -- SHIB

Train window: start -> 2024-02-18

Validation window: 2024-02-19 -> 2024-06-01

Total timesteps: 100000, seed: 42

## PPO vs momentum baseline (same validation window)

|                        |   Cumulative Return |   Sharpe Ratio |   Sortino Ratio |   Max Drawdown |
|:-----------------------|--------------------:|---------------:|----------------:|---------------:|
| PPO (this run)         |           0.0927723 |       0.885375 |        0.802535 |      -0.558302 |
| Momentum (same window) |           0.264303  |       0.999142 |        2.76194  |      -0.669704 |

Mean |action|: 0.6095, std(action): 0.5828 -- see src/eval/reward_hacking_check.py for a full action-collapse check.
