# Phase 6 -- Feature Attribution & Cross-Platform Lead-Lag

## 1. Permutation importance -- DOGE only

Only `models/ppo_DOGE/model.zip` existed as a trained checkpoint when this section was first generated (Phase 4a trained one policy on one asset; see this module's docstring for why the per-asset-class breakdown the dev plan asks for is honestly scoped to this single asset rather than fabricated for asset classes with no trained policy). This is a within-asset-class (small-cap crypto) result only, not a cross-class comparison.

**Update (2026-09-20):** this table has now been extended to all 5 assets -- see `docs/attribution_report_all_assets.md` for the full cross-asset comparison (AMC/GME cleared `src/eval/reward_hacking_check.py` first, per `docs/reward_hacking_findings.md`). This DOGE-only table below is preserved unchanged as the original single-asset record.

Validation window: 2024-02-19 -> 2024-06-01 (21 attention features). Baseline Sharpe on this window: 3.9533.


| feature                       |   mean_action_shift |   std_action_shift |   mean_sharpe_drop |
|:------------------------------|--------------------:|-------------------:|-------------------:|
| search_interest_velocity      |          0.326054   |         0.0221067  |         1.60132    |
| search_interest               |          0.292049   |         0.0305914  |         0.740059   |
| search_interest_acceleration  |          0.233501   |         0.0280405  |         1.7242     |
| pageviews                     |          0.169732   |         0.018421   |        -0.311976   |
| pageviews_velocity            |          0.145308   |         0.0127952  |        -0.0652091  |
| pageviews_acceleration        |          0.122068   |         0.0115417  |         0.212349   |
| edit_count_acceleration       |          0.0924122  |         0.00840122 |         0.45017    |
| edit_count                    |          0.088062   |         0.011869   |        -0.110083   |
| edit_count_velocity           |          0.0823582  |         0.00844704 |        -0.0880032  |
| hn_story_count_acceleration   |          0.0797659  |         0.00276431 |         0.0542889  |
| hn_story_count_velocity       |          0.0622266  |         0.00374985 |        -0.586841   |
| hn_comment_count_acceleration |          0.0550653  |         0.00726431 |        -0.132431   |
| hn_points_acceleration        |          0.0477676  |         0.00575069 |        -0.0797813  |
| hn_story_count                |          0.0363517  |         0.00531794 |        -0.639638   |
| unique_editors                |          0.0252114  |         0.00784507 |        -0.0288821  |
| hn_points_velocity            |          0.0247806  |         0.00553946 |         0.124669   |
| hn_comment_count_velocity     |          0.0237111  |         0.00465985 |        -0.00577801 |
| hn_comment_count              |          0.0171249  |         0.00383402 |        -0.0256514  |
| revert_count                  |          0.0100942  |         0.00506225 |         0.029954   |
| edit_war_ratio                |          0.00985424 |         0.00578673 |         0.0245294  |
| hn_points                     |          0.00610729 |         0.00442673 |        -0.00421673 |


![feature importance](docs/figures/attribution_feature_importance_DOGE.png)


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
