# PPO Training Report -- DOGE

Train window: start -> 2024-02-18

Validation window: 2024-02-19 -> 2024-06-01

Total timesteps: 100000, seed: 42

## PPO vs momentum baseline (same validation window)

|                        |   Cumulative Return |   Sharpe Ratio |   Sortino Ratio |   Max Drawdown |
|:-----------------------|--------------------:|---------------:|----------------:|---------------:|
| PPO (this run)         |             1.42877 |        3.95331 |         5.81484 |      -0.131921 |
| Momentum (same window) |             0.37799 |        1.24986 |         2.22073 |      -0.5109   |

Mean |action|: 0.6003, std(action): 0.6582 -- see src/eval/reward_hacking_check.py for a full action-collapse check.
