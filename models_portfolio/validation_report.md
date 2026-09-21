# Joint Portfolio PPO -- Validation Report

Train window: start -> 2024-03-14

Validation window: 2024-03-15 -> 2024-05-31

Total timesteps: 100000, seed: 42

## Joint policy vs. sum of 5 independent policies (same window)

|                               |   Cumulative Return |   Sharpe Ratio |   Sortino Ratio |   Max Drawdown |
|:------------------------------|--------------------:|---------------:|----------------:|---------------:|
| Joint portfolio policy        |         -0.210893   |         1.3801 |        3.09136  |      -0.663215 |
| Sum of 5 independent policies |         -0.00390288 |         0.2882 |        0.324201 |      -0.366363 |
