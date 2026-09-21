# Phase 6 -- Feature Attribution & Cross-Platform Lead-Lag

## 1. Permutation importance -- SHIB only

Only `models/ppo_SHIB/model.zip` exists as a trained checkpoint (Phase 4a trained one policy on one asset; see this module's docstring for why the per-asset-class breakdown the dev plan asks for is honestly scoped to this single asset rather than fabricated for asset classes with no trained policy). This is a within-asset-class (small-cap crypto) result only, not a cross-class comparison.

Validation window: 2024-02-19 -> 2024-06-01 (21 attention features). Baseline Sharpe on this window: 0.8854.


| feature                       |   mean_action_shift |   std_action_shift |   mean_sharpe_drop |
|:------------------------------|--------------------:|-------------------:|-------------------:|
| search_interest               |         0.71217     |        0.0233737   |        0.378776    |
| pageviews                     |         0.508367    |        0.0278721   |       -0.119256    |
| pageviews_velocity            |         0.303539    |        0.0137478   |       -0.63705     |
| pageviews_acceleration        |         0.272247    |        0.0165572   |        0.0199583   |
| search_interest_acceleration  |         0.114715    |        0.01081     |       -1.53227     |
| search_interest_velocity      |         0.107289    |        0.0107754   |       -0.561468    |
| edit_count_velocity           |         0.0566126   |        0.00737466  |       -0.221684    |
| edit_count_acceleration       |         0.0263958   |        0.00517462  |       -0.0350169   |
| edit_count                    |         0.0254725   |        0.00998245  |       -0.052343    |
| unique_editors                |         0.0228746   |        0.00839424  |        0.0952741   |
| hn_points                     |         0.0028859   |        8.87418e-05 |        0.00341749  |
| hn_story_count                |         0.00230792  |        0.000176825 |        0.00105249  |
| hn_comment_count              |         5.22987e-08 |        6.61744e-24 |       -1.52137e-07 |
| revert_count                  |         5.22987e-08 |        6.61744e-24 |       -1.52137e-07 |
| hn_story_count_velocity       |         5.22987e-08 |        6.61744e-24 |       -1.52137e-07 |
| hn_story_count_acceleration   |         5.22987e-08 |        6.61744e-24 |       -1.52137e-07 |
| hn_points_velocity            |         5.22987e-08 |        6.61744e-24 |       -1.52137e-07 |
| hn_points_acceleration        |         5.22987e-08 |        6.61744e-24 |       -1.52137e-07 |
| hn_comment_count_velocity     |         5.22987e-08 |        6.61744e-24 |       -1.52137e-07 |
| hn_comment_count_acceleration |         5.22987e-08 |        6.61744e-24 |       -1.52137e-07 |
| edit_war_ratio                |         5.22987e-08 |        6.61744e-24 |       -1.52137e-07 |


![feature importance](docs/figures/attribution_feature_importance_SHIB.png)


## 2. Cross-platform lead-lag (raw, pre-derivative signals)

No trained policy is required for this analysis, so it covers every asset with processed features, across all three asset classes, including the two NFT collections (which have no price data and so were excluded from the Phase 5 RL backtest). Reddit-derived raw signals (`post_count`, `comment_count`, `sentiment`) are not used -- Reddit ingestion requires live API credentials that remain a documented, blocked gap (see `configs/assets.yaml`), and none of the processed feature files in this run actually contain those columns. The raw signals actually compared instead: Wikipedia `pageviews`, Google Trends `search_interest`, and Hacker News `hn_story_count`, at lags of -14..+14 days.


| asset             | signal_a        | signal_b        |   best_lag_days |   best_corr |   n_obs | asset_class      |
|:------------------|:----------------|:----------------|----------------:|------------:|--------:|:-----------------|
| AMC               | pageviews       | search_interest |               1 |   0.158758  |     518 | meme_stocks      |
| AMC               | pageviews       | hn_story_count  |              -2 |  -0.0670038 |     518 | meme_stocks      |
| AMC               | search_interest | hn_story_count  |               3 |   0.154227  |     518 | meme_stocks      |
| DOGE              | pageviews       | search_interest |               1 |   0.592022  |     518 | small_cap_crypto |
| DOGE              | pageviews       | hn_story_count  |               1 |   0.530429  |     518 | small_cap_crypto |
| DOGE              | search_interest | hn_story_count  |               0 |   0.309454  |     518 | small_cap_crypto |
| GME               | pageviews       | search_interest |              -9 |   0.28218   |     518 | meme_stocks      |
| GME               | pageviews       | hn_story_count  |               1 |   0.107391  |     518 | meme_stocks      |
| GME               | search_interest | hn_story_count  |              -1 |   0.3131    |     518 | meme_stocks      |
| PEPE              | pageviews       | search_interest |               1 |   0.491     |     518 | small_cap_crypto |
| PEPE              | pageviews       | hn_story_count  |              12 |  -0.087944  |     518 | small_cap_crypto |
| PEPE              | search_interest | hn_story_count  |              -7 |   0.085579  |     518 | small_cap_crypto |
| SHIB              | pageviews       | search_interest |              -1 |   0.876736  |     518 | small_cap_crypto |
| SHIB              | pageviews       | hn_story_count  |              10 |  -0.0354951 |     518 | small_cap_crypto |
| SHIB              | search_interest | hn_story_count  |              11 |  -0.0269635 |     518 | small_cap_crypto |
| boredapeyachtclub | pageviews       | search_interest |               1 |   0.164472  |     518 | nft_collections  |
| boredapeyachtclub | pageviews       | hn_story_count  |              -4 |   0.0660538 |     518 | nft_collections  |
| boredapeyachtclub | search_interest | hn_story_count  |               3 |   0.227112  |     518 | nft_collections  |
| cryptopunks       | pageviews       | search_interest |               0 |   0.586277  |     518 | nft_collections  |
| cryptopunks       | pageviews       | hn_story_count  |              -1 |  -0.0695575 |     518 | nft_collections  |
| cryptopunks       | search_interest | hn_story_count  |               5 |  -0.072905  |     518 | nft_collections  |


(Full table also saved to `docs/figures/leadlag_cross_correlation.csv`.)
