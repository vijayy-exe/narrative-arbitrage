# Narrative Arbitrage: Trading on Attention, Not Price

*Project write-up — Phase 6. Every number below is pulled directly from this
repo's own generated reports (`docs/backtest_report.md`,
`models/ppo_DOGE/validation_report.md`, `docs/attribution_report.md`,
`docs/figures/leadlag_cross_correlation.csv`) or from a command run while
writing this report; none are invented or estimated.*

## 1. Problem Statement

**Core hypothesis:** for narrative-driven assets (meme stocks, small-cap
crypto, NFTs), price is a *lagging* indicator of collective public
attention — the crowd talks, argues, and searches before it buys. If that
is true, an agent that watches only the *derivatives* of attention (how
fast Wikipedia edit activity or Google search interest is accelerating,
whether platforms agree or diverge) should be able to learn a profitable,
risk-aware trading policy **without ever seeing a price chart as input.**

This project builds that agent: a continuous-action PPO policy trading a
basket of meme stocks (AMC, GME), small-cap crypto (DOGE, PEPE, SHIB), and
NFT collections (Bored Ape Yacht Club, CryptoPunks), using only Wikipedia,
Google Trends, and Hacker News attention-derivative features as state,
evaluated against a naive price-momentum baseline and a supervised
(XGBoost) baseline on the same attention features. As the dev plan framed
it going in: a negative result here (attention doesn't lead price) is
still an interesting finding about how information propagates across
platforms — this project isn't scored on profitability alone, and this
report says plainly where the hypothesis held and where it didn't.

## 2. Method

- **State:** first/second derivatives (velocity/acceleration) of Wikipedia
  pageviews, edit count, Google Trends search interest, and Hacker News
  story/point/comment counts, plus edit-war ratio and unique-editor count,
  normalized with an expanding (strictly causal) z-score, concatenated with
  minimal portfolio context (current position, cash, unrealized P&L). See
  `docs/mdp_spec.md` and `docs/feature_dictionary.md` for the full spec.
- **Action:** continuous target position size in `[-1, 1]` (max short to
  max long), sized as a fraction of initial cash so it's comparable across
  assets of very different price levels (`src/env/trading_env.py`).
- **Reward:** realized return net of transaction cost, minus a
  volatility-scaled risk penalty (`R_t = Return_t − λ·Vol_t − Cost(ΔPosition)`,
  `docs/mdp_spec.md` section 3).
- **Algorithm:** PPO (`stable-baselines3`), single-asset training in Phase
  4a, then retrained from scratch per roll per asset for Phase 5's
  walk-forward backtest. **Phase 4b (RecurrentPPO) was deliberately
  skipped** — a scope cut the dev plan itself lists as an acceptable
  fallback under time pressure, not an oversight.
- **Baselines:** naive price-momentum (long/short on trailing 20-day
  return sign — no attention features, no fitting) and a supervised
  XGBoost classifier trained on the *same* attention features as PPO,
  refit fresh every roll.
- **Evaluation:** walk-forward backtesting (`src/eval/walk_forward.py`) —
  6-month train / 1-month validate windows, rolled forward 1 month at a
  time, PPO and the XGBoost baseline retrained from scratch on every roll
  (never fine-tuned forward), with an explicit, tested leakage audit
  (`audit_windows_no_leakage`) run before every evaluation. Metrics:
  Sharpe, Sortino, max drawdown, cumulative return, win rate, average trade
  duration.
- **Attribution (this phase):** permutation importance on the trained PPO
  policy's actions (see section 5), and cross-correlation of raw
  attention signals across platforms and asset classes (section 5).

## 3. Data Pipeline Summary — Including the Honest Gaps

Real data pulled from live public APIs for every asset in `configs/assets.yaml`:
**Wikipedia** (pageviews, edit count, unique editors, revert count),
**Google Trends** (weekly search interest, forward-filled onto the daily
grid), and **Hacker News** (story/point/comment counts via the Algolia
search API). This is the actual, no-auth attention signal set behind every
number in this report.

Three gaps are documented in the codebase and repeated here rather than
glossed over:

1. **No Reddit data.** `src/data_pipeline/reddit.py` is fully implemented
   and would produce `post_count`, `comment_count`, `sentiment`,
   `novelty_score`, and `sentiment_divergence`, but Reddit stopped issuing
   self-serve script-app API credentials as of 2026-09 (manual review, no
   guaranteed approval — see `configs/assets.yaml`'s comment block).
   **Verified for this report:** none of the 7 files in `data/processed/`
   contain a `comment_count` or `sentiment` column. Every Reddit-derived
   claim the original MDP spec envisioned (Reddit comment acceleration,
   sentiment divergence, novelty score) is therefore absent from this run's
   actual state space — the policy trades on Wikipedia + Google Trends +
   Hacker News only.
2. **NFTs excluded from the RL backtest, not from feature collection.**
   `boredapeyachtclub` and `cryptopunks` have real, complete processed
   attention-feature files, but no price data (Phase 1a never implemented
   OpenSea/NFT price ingestion), so `src/agents/common.py::available_assets()`
   excludes them from anything requiring price (training, backtesting).
   They are, however, included in this phase's cross-platform lead-lag
   analysis (section 5), which needs no price data at all.
3. **Phase 4b (RecurrentPPO) skipped.** Only a feed-forward `MlpPolicy` PPO
   was trained. The dev plan's own scope-cut list treats this as an
   acceptable trim, but it does mean the "attention dynamics are
   sequential, so memory should help" hypothesis from the original plan
   was never actually tested — this project's PPO agent has no memory of
   its own beyond the current observation.

## 4. Backtest Results vs. Baselines

Basket-wide mean out-of-sample metrics across all 5 tradeable assets
(AMC, DOGE, GME, PEPE, SHIB) and every walk-forward roll
(`docs/backtest_report.md`):

| method | Cumulative Return | Sharpe Ratio | Sortino Ratio | Max Drawdown | win_rate | avg_trade_duration |
|:---|---:|---:|---:|---:|---:|---:|
| momentum | 0.1711 | **1.1191** | 3.0488 | -0.1939 | 0.4531 | 11.30 |
| ppo | 0.0498 | **0.7120** | 2.8460 | -0.1461 | 0.4739 | 3.43 |
| supervised | -0.0065 | **0.0654** | 1.0649 | -0.2131 | 0.4862 | 3.28 |

Basket-wide, attention-only PPO **does not beat** the naive price-momentum
baseline on mean Sharpe (0.712 vs 1.119), though it clearly beats the
supervised baseline trained on the identical attention features (0.712 vs
0.065) — the same features, fed to a sequential risk-aware policy vs. a
one-shot classifier, produce a very different result. PPO does hold the
smallest max drawdown of the three and by far the shortest average trade
duration (3.4 days vs. momentum's 11.3), consistent with a policy that
reacts to attention-derivative *changes* rather than riding trailing price
trend for weeks at a time.

The basket average, though, hides sharp per-asset variation
(`docs/backtest_report.md`'s per-asset table; rolls per asset: DOGE/SHIB
11, AMC/GME 10, PEPE 6 — shorter due to a shorter aligned history):

| asset | PPO Sharpe | Momentum Sharpe | PPO beats momentum? |
|:---|---:|---:|:---|
| DOGE | 1.7116 | 0.9848 | **yes** |
| SHIB | 1.4555 | -0.2907 | **yes** |
| PEPE | 1.0922 | 1.7840 | no (PPO still positive) |
| AMC | 0.2780 | 2.6768 | no |
| GME | -0.9995 | 0.8612 | no |

PPO beats momentum on **DOGE and SHIB only** — both small-cap crypto — and
loses to momentum on AMC, GME (meme stocks) and PEPE (crypto, but the
weakest-quality Wikipedia signal in the basket — see section 5's caveat on
PEPE's proxy article). On GME specifically, PPO is not just worse than
momentum, it has a *negative* mean Sharpe across its 10 rolls.

For context, Phase 4a's original single-window DOGE result (100k
timesteps, one train/val split, not walk-forward) showed PPO at Sharpe
**3.95** vs. momentum's 1.25 on the same window
(`models/ppo_DOGE/validation_report.md`) — dramatically higher than the
walk-forward average of 1.71. This is exactly the gap Phase 5 was designed
to catch: the single-window number was optimistic, not fabricated, but not
representative of performance across 11 different rolled train/validate
windows either. A rerun of `src/eval/reward_hacking_check.py` on that same
DOGE checkpoint (mean action 0.236, std 0.658, only 9.7% of actions
near-flat) confirms the policy is not a degenerate always-long/always-flat
collapse propping up that one number — it genuinely varies with input —
but "not collapsed" is a floor, not proof the edge generalizes, and the
walk-forward results show it only partially does.

## 5. Attribution Findings

### 5a. Permutation importance — scoped honestly to DOGE only

**What was and wasn't run, stated plainly:** at the time this section was
first written, only `models/ppo_DOGE/model.zip` existed as a trained,
persisted checkpoint. Phase 4a trained one policy on one asset; Phase 5's
walk-forward retrains a fresh PPO per roll per asset but never saves those
checkpoints (by design — the walk-forward harness measures generalization
across rolls, not a single deployable model). Rather than train new
throwaway per-asset-class policies just to populate a chart with three
bars, this analysis was scoped to the one policy that actually existed,
and clearly labeled: **this was a within-crypto result, not a
cross-asset-class comparison.** Training additional lightweight policies
for a meme-stock and an NFT representative was considered and rejected as
the more honest choice under the time available at that point — see this
report's discussion of what would be needed to close this gap (section 6).

**Update (2026-09-20):** as of 2026-09-15, `models/ppo_{AMC,GME,PEPE,SHIB}/model.zip`
were also trained (see `docs/CURRENT_STATUS.md`), so the "only DOGE has a
checkpoint" justification above is no longer factually current. AMC and
GME subsequently cleared a reward-hacking audit
(`docs/reward_hacking_findings.md` — both are genuine, non-collapsed
policies), unblocking exactly the extension this update originally
deferred. **That extension is now done**: `docs/
attribution_report_all_assets.md` runs the same permutation-importance
method against all 5 assets. Headline finding: small-cap crypto (DOGE,
PEPE) relies most on Google Trends *velocity* — matching this project's
core "derivatives, not levels" hypothesis — while meme stocks (AMC, GME)
rely most on Wikipedia *acceleration* instead, and **SHIB is a clean
exception to the whole hypothesis**, relying most on raw `search_interest`
*level* (0.712 mean action-shift, more than double any other
feature/asset pair in the basket) rather than any derivative. See that
document for the full breakdown and its own caveats (still single-window,
still single-seed, same limitations as this section's original DOGE
analysis).

Method: one deterministic rollout on DOGE's validation window
(2024-02-19 → 2024-06-01, 103 steps) produces a baseline observation
matrix and action sequence. For each of the 21 attention-feature columns,
the column is shuffled in time (5 repeats, averaged), the policy is
re-queried on the shuffled observations, and the resulting counterfactual
action sequence is *actually replayed* through a fresh episode of the
trading environment to get a real resulting Sharpe ratio — not just a
change in the raw action number. Full table: `docs/attribution_report.md`;
chart: `docs/figures/attribution_feature_importance_DOGE.png`.

Top 6 by mean action shift under permutation:

| feature | mean \|action shift\| | mean Sharpe drop |
|:---|---:|---:|
| search_interest_velocity | 0.326 | 1.601 |
| search_interest | 0.292 | 0.740 |
| search_interest_acceleration | 0.234 | 1.724 |
| pageviews | 0.170 | -0.312 |
| pageviews_velocity | 0.145 | -0.065 |
| pageviews_acceleration | 0.122 | 0.212 |

Google Trends' `search_interest` (level, velocity, and acceleration) are
the three most important features by a wide margin — roughly 2x the
action-shift of the next-strongest group (Wikipedia `pageviews` features),
which in turn dominate Wikipedia `edit_count` features, which dominate
Hacker News features (`hn_story_count`/`hn_points`/`hn_comment_count`,
mean action shift 0.006–0.080) and `edit_war_ratio`/`revert_count`
(weakest, ~0.01). This independently corroborates a finding already
documented elsewhere in this codebase (`src/data_pipeline/features.py`'s
`recompute_derivatives_on_index` docstring), which found
`search_interest_velocity` to have the strongest raw correlation with
next-day return of any feature (0.23 for DOGE, 0.38 for SHIB) — two
different methods (raw return-correlation vs. permutation importance on
the trained policy's actions) agree on the same feature being dominant for
this asset. The `mean_sharpe_drop` column is noisier (small validation
window, only 103 days — a few features even show a *negative* mean Sharpe
drop, i.e. permuting them happened to help on this particular window by
chance) and should not be over-read feature-by-feature, but its top
entries broadly agree with the action-shift ranking's top 3.

### 5b. Cross-platform lead-lag — all three asset classes

This half of the analysis needs no trained policy, so unlike 5a it
honestly covers all 7 assets with processed features, across all three
classes, including both NFT collections. Reddit-derived raw signals
(`post_count`, `comment_count`, `sentiment`) are **not used** — see
section 3's gap #1; they simply don't exist in this run's data. The raw,
pre-derivative signals actually compared: Wikipedia `pageviews`, Google
Trends `search_interest`, and Hacker News `hn_story_count`, at day lags of
-14 to +14 (full table: `docs/figures/leadlag_cross_correlation.csv`).

**Pageviews vs. search_interest, by asset class** (the strongest and most
consistently-signed pair):

| asset | asset class | best \|corr\| | best lag (days) |
|:---|:---|---:|---:|
| SHIB | small_cap_crypto | 0.877 | -1 (pageviews leads) |
| DOGE | small_cap_crypto | 0.592 | +1 (search_interest leads) |
| cryptopunks | nft_collections | 0.586 | 0 (contemporaneous) |
| PEPE | small_cap_crypto | 0.491 | +1 (search_interest leads) |
| GME | meme_stocks | 0.282 | -9 |
| boredapeyachtclub | nft_collections | 0.164 | +1 (search_interest leads) |
| AMC | meme_stocks | 0.159 | +1 (search_interest leads) |

Two honest findings, one confirming a class effect and one complicating it:

- **Correlation strength does split by class, on average.** The three
  small-cap crypto assets average |corr| ≈ 0.65 (all three ≥ 0.49), while
  the two meme stocks average ≈ 0.22 (both ≤ 0.28) — crypto attention
  genuinely couples Wikipedia and Google Trends activity more tightly than
  meme-stock attention does, in this data.
- **NFTs split down the middle rather than forming a clean third
  cluster.** CryptoPunks (0.586) looks like a crypto asset; Bored Ape
  (0.164) looks like a meme stock. "NFT" is not a coherent bucket for this
  particular relationship — it behaves like whichever community's online
  discussion pattern it happens to resemble, not like a distinct asset
  class.
- **Lead/lag direction is not stable even within one class.** DOGE's
  strongest lag is +1 (Google Trends search interest leads Wikipedia
  pageviews by a day), while SHIB's is -1 (the opposite — Wikipedia leads
  Trends). Both are "small_cap_crypto." There is no single answer to "does
  Wikipedia lead Reddit/Trends for crypto" in this data — the correlation
  is real and class-correlated in strength, but which platform leads
  flips asset by asset.

`hn_story_count`'s cross-correlations with either other signal are weak
(mostly |corr| < 0.16) and sign-inconsistent across nearly every asset —
consistent with `docs/feature_dictionary.md`'s own documented caveat that
Hacker News volume for these niche tickers is genuinely sparse (most days
are a real zero, not a fetch gap), so there's little signal for a
cross-correlation to find.

## 6. Honest Discussion: Where the Hypothesis Held vs. Failed

The core hypothesis — "attention leads price, so an attention-only agent
can beat a price-only baseline" — **held for DOGE and SHIB, and failed for
AMC, GME, and (on Sharpe, though not on raw sign) PEPE.**

**Where it held:** on DOGE and SHIB, PPO trained on nothing but Wikipedia/
Trends/HN derivatives beat a price-momentum baseline out-of-sample across
11 independently-retrained walk-forward rolls each — not a lucky single
window. Both are also exactly the two assets whose `pageviews` vs.
`search_interest` cross-correlation was strongest (0.59 and 0.88) in
section 5b — suggestive, though this report does not claim to have proven
a causal link between "how tightly an asset's attention platforms move
together" and "how tradeable its attention derivatives are," only that the
two facts co-occur for these two assets.

**Where it failed:** on AMC and GME — the two meme stocks — momentum beat
PPO by a wide margin (2.68 and 0.86 vs. PPO's 0.28 and -0.99), and these
are exactly the two assets with the weakest pageviews/search_interest
cross-correlation (0.16, 0.28) in section 5b. On PEPE, PPO stayed
Sharpe-positive (1.09) but still lost to momentum (1.78) — and PEPE
carries a known, documented data-quality caveat unrelated to the
hypothesis itself: there is no dedicated Wikipedia article for the Pepe
*cryptocurrency*, so its Wikipedia signal (`Pepe the Frog`) reflects
general meme attention, not coin-specific attention (`configs/assets.yaml`).
It is plausible PEPE's result is at least partly a data-quality artifact
rather than pure evidence against the hypothesis, but that is a caveat,
not an excuse — the walk-forward number is what it is.

**A complication the supervised baseline exposes:** if attention
derivatives alone were straightforwardly predictive, the supervised
XGBoost baseline — trained on the *identical* features PPO sees — should
also show an edge. It doesn't (basket Sharpe 0.065, barely above zero,
and negative Cumulative Return). This means the story is not simply
"attention data contains a directly learnable predictive signal price
data lacks" — it's specifically that a sequential, risk-aware, continuous-
position-sizing policy extracts value from these features that a one-shot
classifier making a next-day up/down call does not. That's a meaningfully
narrower and more interesting claim than "attention beats price," and it's
the one this project's actual numbers support.

**What would be needed to fully answer the dev plan's original ask:**
a true per-asset-class permutation-importance comparison would need at
least one trained PPO policy for a meme-stock and an NFT-representative
asset (impossible for NFTs specifically without first solving the missing
NFT price-data gap in section 3). Given the time available, this report
chose to be honest about running the cross-platform lead-lag analysis in
full (it required no new training) rather than fabricate or hastily train
under-converged policies just to fill out a three-class attribution chart.
That is a real, acknowledged gap in this write-up, not a hidden one.

## 7. RL Upgrade Path (2026-09-20) — Risk Sensitivity, Architecture, Hyperparameters

Following `narrative-arbitrage-context-upgrade.md`'s diagnosis that this
project's RL was "vanilla and shallow" (plain PPO, single seed, flat MLP,
expected-return-only objective, five independent single-asset silos), a
sequenced upgrade path was implemented and evaluated —
`narrative-arbitrage-implementation-upgrade.md`. **Methodological note,
stated up front so it isn't lost in the numbers below: every result in this
section is single-validation-window (Phase-4a-style: 100k timesteps, one
train/validate split, seed 42 unless multi-seeded), matching the protocol
the original per-asset checkpoints (section 4's single-window DOGE citation)
used — NOT section 4's walk-forward protocol.** Section 4's basket table
remains the only walk-forward-validated result in this report. Everything
below should be read as "a real, reproducible effect on this specific
window," not yet re-validated at walk-forward rigor — each subsection's
own doc repeats this caveat and names walk-forward re-validation as the top
follow-up.

**0. Known-issues resolution** (`docs/CURRENT_STATUS.md`'s 2026-09-20
addendum, `docs/reward_hacking_findings.md`): the DOGE-only staleness noted
in section 5a above was fixed (tests + this report updated); AMC and GME
were audited for reward hacking/policy collapse and **cleared** — both are
genuine (if bad) learned policies, not degenerate ones. GME's failure
mode is a clean directional miss (net short into a +72.6% rally); AMC's
looks like overconfident sizing/overtrading. A Reddit-replacement
evaluation (StockTwits, 4chan) found 4chan works live but has no historical
archive — the historical `comment_count`/`sentiment`/`novelty_score` gap
(section 3, gap #1) is now understood to be **permanent**, not pending.

**1. Risk-sensitive (CVaR) objective** (`docs/risk_sensitivity_ablation.md`):
a genuine quantile-critic CVaR-PPO, trained on all 5 assets. Improves
Sharpe/Sortino on 4 of 5 assets (GME flips from −3.82 to +0.15; AMC
improves from −2.03 to −0.93) but **worsens cumulative return and max
drawdown on all 5 assets, no exceptions** — the mechanism appears to reward
*consistency* of exposure (fewer direction flips → smoother return series)
rather than correctness of the directional call. Not a strict upgrade over
the original objective; a genuine, documented tradeoff.

**2+3. Architecture ablation — MLP vs. LSTM vs. Transformer**
(`docs/architecture_ablation.md`): a Transformer temporal-attention encoder
(`src/agents/attention_encoder.py`) and `RecurrentPPO`/LSTM
(`sb3-contrib`) were each trained on all 5 assets and compared against the
original flat-MLP baseline. **LSTM wins on every basket-average metric**
(Sharpe +1.161, Sortino +3.675, max DD −0.647, and the only architecture
with a positive average cumulative return, +0.054) — beating both the MLP
baseline and the CVaR objective above outright, on average. This is a real
answer to the original dev plan's cut Phase 4b hypothesis ("does
sequential memory help?") — yes, more than either alternative tried. **But
no architecture wins per-asset**: DOGE is still best served by the plain
MLP, AMC by the Transformer, GME by the LSTM — a per-asset architecture
choice would likely beat any single basket-wide one.

**4. Hyperparameter search + multi-seed reporting**
(`docs/hparam_search_results.md`, `docs/multiseed_report.md`): a 20-trial
Optuna sweep (learning rate, entropy coefficient, GAE lambda, LSTM hidden
size) on DOGE — LSTM being the architecture ablation's winner, DOGE being
the dev prompt's named "most stable" search asset (a different claim from
"the asset LSTM helps most," stated plainly in both docs) — found
`lstm_hidden_size=128, learning_rate≈4.6e-4, ent_coef≈3.7e-4,
gae_lambda≈0.835` as the best config. Retrained across 5 seeds at the full
100k-timestep budget: **Sharpe 1.719 ± 1.798, Sortino 3.840 ± 2.979** — the
standard deviation is nearly as large as the mean, and one seed (of five,
otherwise identical) came out Sharpe **−1.746** while the other four
clustered at 2.24–3.51. **This is this section's single most important
finding, and it reframes every other single-seed number above and in
Upgrades 1-3**: this project's seed=42 point estimates were never as solid
as a single number implied, and there was no way to know that without
this run. It does not overturn the architecture-ablation LSTM finding for
DOGE specifically (seed 42 happened to land in the good cluster), but
every cross-architecture/cross-objective comparison in section 7 above
should now be read as "true for that one seed," not "true in general."

**5. Walk-forward re-validation of Upgrades 1-3**
(`docs/walk_forward_upgrade_validation.md`) — the single most important
result of this entire upgrade path, and it is a sobering one. All three
new RL variants were retrained from scratch on the exact same
walk-forward protocol as section 4's original PPO/momentum/supervised
numbers (48 rolls, 15k timesteps/roll). **None of them beat the original
plain-MLP PPO's walk-forward Sharpe (0.712):**

| Method | Walk-forward Sharpe |
|---|---:|
| Momentum (baseline) | 1.119 |
| **Original PPO / plain MLP (Upgrade 0)** | **0.712** |
| Transformer-PPO (Upgrade 2) | 0.512 |
| LSTM/RecurrentPPO (Upgrade 3) | 0.451 |
| CVaR-PPO (Upgrade 1) | 0.110 |
| Supervised / XGBoost (baseline) | 0.065 |

**Every single-window "win" reported above either shrinks or reverses.**
CVaR's single-window claim (Sharpe/Sortino improve on 4/5 assets) fully
reverses — walk-forward CVaR is the *worst* of the three new architectures
tried. The architecture ablation's AMC/Transformer single-window win
(+0.351, "the only positive number this asset has produced") does not
survive: Transformer's walk-forward AMC Sharpe is −0.164, statistically
indistinguishable from LSTM's −0.164. **The one genuine win in all 15
(asset × new-architecture) cells tried is LSTM on GME** (seed 42: −1.00 →
+1.38) — though a 3-seed check (42, 100, 101: 1.38, 1.92, −0.62) found this
carries the same seed-to-seed variance seen everywhere else in this
project. The mean across those 3 seeds (+0.89) is still clearly better
than original PPO's −1.00, so "a real average improvement" holds, but
"reliably fixed" does not — one bad seed still flips it negative. See
`docs/walk_forward_upgrade_validation.md`'s seed-robustness section for
the full check.

**Net effect on this report's core finding:** section 6's walk-forward
conclusion ("the hypothesis held for DOGE/SHIB, failed for AMC/GME/PEPE")
is **not overturned**, and is if anything reinforced: none of Upgrades 1-3
found a way to make AMC, GME (beyond the one LSTM exception above), or
PEPE's attention-only edge walk-forward-positive. What section 4's original
single-window-vs-walk-forward gap warned about (DOGE: 3.95 single-window
vs. 1.71 walk-forward) turned out to generalize to every subsequent
upgrade tried on top of it, not just the original PPO run — the lesson
this project already knew in the abstract (`docs/CURRENT_STATUS.md`,
before any of Upgrades 1-4 existed) turned out to be exactly this
consequential in practice.

## 8. Future Work

**Done since this section was first written (2026-09-20, same day):**
- **Walk-forward re-validation of Upgrades 1-3** — done, see the new
  subsection above and `docs/walk_forward_upgrade_validation.md`. Result:
  none of the three beat the original PPO basket-wide; LSTM-on-GME is the
  one confirmed exception. (Not separately re-run: the SPECIFIC tuned
  hyperparameter config from Upgrade 4 — the walk-forward LSTM numbers use
  RecurrentPPO's untuned defaults, matching the architecture ablation's own
  protocol, not the sweep's winning learning-rate/entropy/gae-lambda/
  hidden-size combination. Whether the tuned config changes GME's already-
  strong walk-forward LSTM result, or helps any other asset catch up to
  it, is a small, cheap, still-open follow-up.)
- **Multi-seed reporting extended to all 5 assets** for the tuned LSTM
  config — done, see `docs/multiseed_report.md`. Every asset shows at
  least one seed that flips sign or is a dramatic outlier; this is a
  basket-wide property of the config, not a DOGE-specific fluke.
- **Attribution extended to all 5 assets** — done, see `docs/
  attribution_report_all_assets.md` (§5a's update note above has the
  headline finding: crypto relies on Trends velocity, meme stocks on
  Wikipedia acceleration, and SHIB is a clean exception relying on raw
  level instead of any derivative).

**Still open, in order of value:**
1. **Retrofit multi-seed reporting to the CVaR and Transformer ablations**,
   and to the LSTM/Transformer/CVaR walk-forward runs above (all of which
   are still single-seed=42). Given the seed-variance magnitude already
   found for the tuned LSTM config, it's plausible — not yet confirmed —
   that CVaR's or Transformer's walk-forward ranking could shift with
   different seeds, though the gap to original PPO (0.712) is large enough
   for CVaR (0.110) that seed variance closing it entirely seems unlikely
   based on the variance magnitudes seen so far.
2. **Investigate LSTM-on-GME specifically** — the one clear, walk-forward-
   confirmed win in this entire upgrade path deserves closer study (does it
   hold with multi-seeding? with tuned hyperparameters? what is the policy
   actually doing differently?) rather than being noted once and left.

**Reach items, deliberately not implemented — described here as future
work rather than built, per `narrative-arbitrage-implementation-upgrade.md`'s
own stopping-point guidance** ("Upgrades 5-6 — only if time allows; fine to
write up as future work otherwise"):

- **Upgrade 5 — joint multi-asset portfolio policy.** All architectures
  evaluated in this report (including LSTM, the walk-forward winner for
  GME specifically) are still five
  independent single-asset policies. A single policy allocating across the
  whole basket jointly (shared encoder, per-asset action heads or a
  portfolio-level action space, portfolio-level reward) could learn
  cross-asset correlation/diversification behavior (e.g. "reduce size in
  both DOGE and SHIB when their attention spikes together because they're
  correlated") that five separate policies structurally cannot express.
  The interesting open question: would joint training reduce drawdown
  through implicit diversification even without improving raw return?
- **Upgrade 6 — offline pretraining / Decision Transformer warm-start.**
  This report's own data shows real single-window-vs-walk-forward
  instability (section 4: DOGE's 3.95 single-window Sharpe vs. 1.71
  walk-forward average) consistent with a data-starved policy learning
  purely from scratch via online exploration. Behavior-cloning
  pretraining from the momentum/supervised baselines' trajectories (both
  already exist, section 2/4), or a minimal Decision Transformer trained
  on the walk-forward backtest's own historical trajectories, are two
  concrete, buildable approaches to test whether pretraining narrows that
  instability gap — neither attempted in this report.

## 9. Summary

An attention-only PPO policy, evaluated with a leakage-audited walk-forward
protocol against a price-only momentum baseline and a same-features
supervised baseline, beats momentum specifically on DOGE and SHIB (two of
five tradeable assets), loses to it on AMC, GME, and PEPE, and beats the
supervised baseline everywhere in basket aggregate. Permutation importance
on the one trained policy that exists (DOGE) shows Google Trends
`search_interest` derivatives are by far its most-relied-on input,
corroborating an independent correlation-based finding already in this
codebase. Cross-platform lead-lag analysis across all three asset classes
(the one part of this phase that didn't require a trained model) shows
Wikipedia/Google-Trends correlation strength genuinely differs by class
(crypto tighter than meme stocks on average) but lead/lag *direction* does
not — and the NFT class splits into one asset that behaves like crypto and
one that behaves like a meme stock rather than forming its own pattern.
The headline result is neither "the hypothesis is confirmed" nor "the
hypothesis is false" — it is asset-specific, and this report has tried to
say exactly which assets and why, rather than average the disagreement
away.

**Section 7's RL upgrade path adds a genuinely important lesson, but not
the one it initially appeared to.** A risk-sensitive objective and two
alternative architectures (Transformer, LSTM) were each evaluated
single-window first, where all three looked like clear improvements —
CVaR fixing 4 of 5 assets' Sharpe, LSTM winning on every basket-average
metric, Transformer producing AMC's only positive result under any
config tried. **Walk-forward re-validation reversed nearly all of it.**
None of the three beats the original plain-MLP PPO's walk-forward Sharpe
(0.712) at the basket level — CVaR (0.110) is the worst RL variant tried,
LSTM (0.451) and Transformer (0.512) both trail the original. The one
exception that shows a real average improvement is **LSTM specifically on
GME** (−1.00 → +0.89 mean across 3 seeds, largest improvement found
anywhere in this project on average) — though even this carries
substantial seed-to-seed variance (one of three seeds tested was
negative), so "reliably fixed" overstates it; "a real, if noisy, average
improvement" is the honest claim. The project's own
bar for a trustworthy result — established in section 4, when the
original single-window DOGE number (3.95) turned out to be an unreliable
guide to its walk-forward reality (1.71) — turned out to apply just as
much force to every subsequent architectural improvement layered on top,
not just to the original baseline. That is this report's final, and most
important, methodological finding: **single-window ablation, in this
project, consistently overstated real generalization, across every
architecture and objective tried, without exception** — the discipline
to re-validate before trusting an improvement is not optional scaffolding
this project could have skipped.
