# PPO Training Report -- PEPE

Train window: start -> 2024-03-14

Validation window: 2024-03-15 -> 2024-06-01

Total timesteps: 100000, seed: 42

## PPO vs momentum baseline (same validation window)

|                        |   Cumulative Return |   Sharpe Ratio |   Sortino Ratio |   Max Drawdown |
|:-----------------------|--------------------:|---------------:|----------------:|---------------:|
| PPO (this run)         |          -0.0748453 |       0.373407 |        0.631345 |       -0.58101 |
| Momentum (same window) |          -0.0942907 |       0.415761 |        0.882481 |       -0.55681 |

Mean |action|: 0.8221, std(action): 0.5860 -- see src/eval/reward_hacking_check.py for a full action-collapse check.
