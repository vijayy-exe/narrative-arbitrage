# Supervised Baseline Report

XGBoost classifier predicting next-day return direction from attention-derivative features only (never price). Time-series 80/20 train/test split per asset, no shuffling. This is a baseline to beat, not the project's actual method (RL).

## Per-asset date boundaries and metrics

|      |   Cumulative Return |   Sharpe Ratio |   Sortino Ratio |   Max Drawdown |   Accuracy |      AUC | Train Start   | Train End   | Test Start   | Test End   |
|:-----|--------------------:|---------------:|----------------:|---------------:|-----------:|---------:|:--------------|:------------|:-------------|:-----------|
| GME  |           -0.913146 |      -1.99468  |       -1.8701   |      -0.929712 |   0.408451 | 0.381746 | 2023-01-05    | 2024-02-16  | 2024-02-20   | 2024-05-30 |
| AMC  |           -0.831999 |      -1.58654  |       -1.30881  |      -0.869141 |   0.521127 | 0.497967 | 2023-01-05    | 2024-02-16  | 2024-02-20   | 2024-05-30 |
| PEPE |           -0.742452 |      -2.49697  |       -3.62911  |      -0.818706 |   0.405063 | 0.471042 | 2023-05-07    | 2024-03-13  | 2024-03-14   | 2024-05-31 |
| DOGE |            0.49454  |       1.45179  |        2.31445  |      -0.540405 |   0.543689 | 0.514361 | 2023-01-03    | 2024-02-18  | 2024-02-19   | 2024-05-31 |
| SHIB |           -0.590596 |      -0.531537 |       -0.497406 |      -0.782112 |   0.553398 | 0.534848 | 2023-01-03    | 2024-02-18  | 2024-02-19   | 2024-05-31 |
