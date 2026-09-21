# Momentum Baseline Report

Naive price-momentum strategy: long if trailing-window return is positive, short otherwise. Uses price only, no attention features -- this is the sanity-check floor, not the project's actual method.

Window: 20 trading days

## Per-asset date range and metrics

|      |   Cumulative Return |   Sharpe Ratio |   Sortino Ratio |   Max Drawdown | Start Date   | End Date   |   N Days |
|:-----|--------------------:|---------------:|----------------:|---------------:|:-------------|:-----------|---------:|
| GME  |           -0.345941 |       0.289004 |        0.450497 |      -0.717776 | 2023-01-04   | 2024-05-31 |      354 |
| AMC  |            1.42649  |       1.07908  |        1.75679  |      -0.801008 | 2023-01-04   | 2024-05-31 |      354 |
| PEPE |            5.90347  |       1.54865  |        2.85932  |      -0.74094  | 2023-05-06   | 2024-06-01 |      393 |
| DOGE |            0.216087 |       0.469081 |        0.720922 |      -0.5109   | 2023-01-02   | 2024-06-01 |      517 |
| SHIB |           -0.244414 |       0.191204 |        0.387709 |      -0.669704 | 2023-01-02   | 2024-06-01 |      517 |
