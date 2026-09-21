# Phase 6 -- Feature Attribution & Cross-Platform Lead-Lag

## 1. Permutation importance -- PEPE only

Only `models/ppo_PEPE/model.zip` exists as a trained checkpoint (Phase 4a trained one policy on one asset; see this module's docstring for why the per-asset-class breakdown the dev plan asks for is honestly scoped to this single asset rather than fabricated for asset classes with no trained policy). This is a within-asset-class (small-cap crypto) result only, not a cross-class comparison.

Validation window: 2024-03-15 -> 2024-06-01 (21 attention features). Baseline Sharpe on this window: 0.3734.


| feature                       |   mean_action_shift |   std_action_shift |   mean_sharpe_drop |
|:------------------------------|--------------------:|-------------------:|-------------------:|
| search_interest_velocity      |         0.27716     |        0.0202976   |        0.495029    |
| pageviews                     |         0.194631    |        0.0131153   |        0.759509    |
| hn_story_count_acceleration   |         0.193805    |        0.0317667   |       -0.201148    |
| search_interest_acceleration  |         0.172534    |        0.039184    |       -0.412378    |
| pageviews_velocity            |         0.163272    |        0.0273379   |       -0.163569    |
| search_interest               |         0.153874    |        0.0122738   |       -0.0401341   |
| hn_comment_count_acceleration |         0.146878    |        0.0225914   |        0.113011    |
| hn_story_count_velocity       |         0.115995    |        0.0170556   |       -0.803656    |
| pageviews_acceleration        |         0.111561    |        0.00526945  |       -0.583418    |
| hn_points_velocity            |         0.0725872   |        0.00619443  |       -0.40477     |
| hn_points                     |         0.0655927   |        0.00928411  |        0.059294    |
| hn_story_count                |         0.0643491   |        0.020177    |       -0.0151194   |
| hn_points_acceleration        |         0.044214    |        0.00685947  |       -0.207965    |
| hn_comment_count_velocity     |         0.03891     |        0.0133046   |       -0.160315    |
| hn_comment_count              |         0.0245143   |        0.0106564   |       -0.135276    |
| edit_count                    |         0.00367095  |        0.000301772 |        0.013227    |
| unique_editors                |         0.0023573   |        0.000205742 |        0.0121754   |
| revert_count                  |         0.00129542  |        0.000133189 |        0.00489223  |
| edit_war_ratio                |         0.000892207 |        0.000109305 |        0.00481397  |
| edit_count_acceleration       |         4.5182e-05  |        7.05872e-06 |       -0.000289948 |
| edit_count_velocity           |         4.50856e-08 |        0           |       -9.46236e-08 |


![feature importance](docs/figures/attribution_feature_importance_PEPE.png)


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
