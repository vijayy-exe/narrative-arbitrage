# Phase 6 -- Feature Attribution & Cross-Platform Lead-Lag

## 1. Permutation importance -- GME only

Only `models/ppo_GME/model.zip` exists as a trained checkpoint (Phase 4a trained one policy on one asset; see this module's docstring for why the per-asset-class breakdown the dev plan asks for is honestly scoped to this single asset rather than fabricated for asset classes with no trained policy). This is a within-asset-class (small-cap crypto) result only, not a cross-class comparison.

Validation window: 2024-02-21 -> 2024-05-31 (21 attention features). Baseline Sharpe on this window: -3.8162.


| feature                       |   mean_action_shift |   std_action_shift |   mean_sharpe_drop |
|:------------------------------|--------------------:|-------------------:|-------------------:|
| pageviews_acceleration        |         0.457798    |        0.0421565   |       -3.47987     |
| pageviews_velocity            |         0.380089    |        0.0391083   |       -4.53317     |
| hn_comment_count_velocity     |         0.246524    |        0.0232591   |       -5.25713     |
| edit_count_acceleration       |         0.24434     |        0.0450761   |       -4.22867     |
| hn_comment_count_acceleration |         0.226964    |        0.0342997   |       -4.4958      |
| hn_story_count_acceleration   |         0.199962    |        0.0193525   |       -4.27239     |
| edit_count_velocity           |         0.182895    |        0.027988    |       -3.22573     |
| pageviews                     |         0.155749    |        0.0146167   |       -2.83986     |
| search_interest               |         0.130092    |        0.0161734   |       -3.17231     |
| hn_points_acceleration        |         0.12896     |        0.018115    |       -4.19035     |
| search_interest_acceleration  |         0.115416    |        0.00940033  |       -4.36298     |
| hn_story_count                |         0.103971    |        0.021104    |       -3.6871      |
| unique_editors                |         0.0994758   |        0.0125898   |       -3.46037     |
| hn_story_count_velocity       |         0.0937592   |        0.0192369   |       -2.84159     |
| hn_points                     |         0.0888683   |        0.00918239  |       -3.79512     |
| hn_points_velocity            |         0.0810567   |        0.0181813   |       -4.10298     |
| search_interest_velocity      |         0.0637021   |        0.015973    |       -4.95648     |
| hn_comment_count              |         0.0564928   |        0.0128091   |       -3.63938     |
| edit_count                    |         0.0487093   |        0.0139354   |       -2.26221     |
| edit_war_ratio                |         0.00108555  |        0.000141736 |        0.0122833   |
| revert_count                  |         0.000695732 |        0.000131698 |        0.000466833 |


![feature importance](docs/figures/attribution_feature_importance_GME.png)


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
