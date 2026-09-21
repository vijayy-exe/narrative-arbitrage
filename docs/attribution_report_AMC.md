# Phase 6 -- Feature Attribution & Cross-Platform Lead-Lag

## 1. Permutation importance -- AMC only

Only `models/ppo_AMC/model.zip` exists as a trained checkpoint (Phase 4a trained one policy on one asset; see this module's docstring for why the per-asset-class breakdown the dev plan asks for is honestly scoped to this single asset rather than fabricated for asset classes with no trained policy). This is a within-asset-class (small-cap crypto) result only, not a cross-class comparison.

Validation window: 2024-02-21 -> 2024-05-31 (21 attention features). Baseline Sharpe on this window: -2.0298.


| feature                       |   mean_action_shift |   std_action_shift |   mean_sharpe_drop |
|:------------------------------|--------------------:|-------------------:|-------------------:|
| edit_count_acceleration       |         0.335706    |        0.0386641   |        -1.39738    |
| pageviews_velocity            |         0.324321    |        0.0289919   |        -1.12673    |
| search_interest               |         0.228845    |        0.0169633   |        -1.93338    |
| hn_story_count_acceleration   |         0.194539    |        0.00903178  |        -1.29365    |
| search_interest_acceleration  |         0.1806      |        0.0242287   |        -0.755145   |
| pageviews                     |         0.177523    |        0.0225033   |        -1.988      |
| pageviews_acceleration        |         0.171769    |        0.0188282   |        -1.13898    |
| edit_count                    |         0.166535    |        0.0340021   |        -1.23066    |
| edit_count_velocity           |         0.151326    |        0.0248807   |        -0.895645   |
| search_interest_velocity      |         0.13981     |        0.0242219   |        -1.26621    |
| unique_editors                |         0.0831026   |        0.0078905   |        -0.0575431  |
| edit_war_ratio                |         0.0665584   |        0.0162027   |        -0.106158   |
| hn_story_count_velocity       |         0.0413606   |        0.0121953   |        -0.0932398  |
| hn_story_count                |         0.0361401   |        0.00928229  |        -0.259742   |
| revert_count                  |         0.0161621   |        0.00939839  |        -0.337505   |
| hn_comment_count              |         0.00529054  |        0.00188779  |         0.0582406  |
| hn_points                     |         0.00165543  |        0.000648514 |         0.0100723  |
| hn_comment_count_acceleration |         0.00122928  |        0.000347479 |         0.00354796 |
| hn_points_acceleration        |         0.000756275 |        0.000278823 |        -0.0516933  |
| hn_comment_count_velocity     |         0.000710179 |        8.58717e-05 |         0.0234322  |
| hn_points_velocity            |         0.000395257 |        0.000326864 |        -0.0176746  |


![feature importance](docs/figures/attribution_feature_importance_AMC.png)


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
