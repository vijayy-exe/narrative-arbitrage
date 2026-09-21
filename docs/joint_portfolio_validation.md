# Joint Multi-Asset Portfolio Policy — Upgrade 5

`docs/final_report.md` section 8 named this as a reach item: every
architecture/objective evaluated in this project (plain PPO, CVaR, LSTM,
Transformer) is five independent single-asset policies, with no way to
express "DOGE and SHIB's attention is spiking together, reduce size in
both because they're correlated." `src/env/multi_asset_env.py`'s
`MultiAssetTradingEnv` closes that gap: one environment whose observation
(all 5 assets' features + portfolio context, concatenated) and action (a
5-dimensional vector of target positions) span the whole basket jointly,
so a single PPO policy can learn genuine cross-asset structure. Code:
`src/agents/train_portfolio_ppo.py`.

**Date alignment cost, stated up front:** a joint decision needs every
asset's state on the same calendar day, so the environment is restricted
to the INTERSECTION of trading days across all 5 assets — 270 days
(2023-05-05 → 2024-05-31), bounded by GME/AMC's weekday-only stock
calendar and PEPE's later start (no dedicated Wikipedia article history
before May 2023). This is meaningfully shorter than e.g. DOGE's own
518-day individual range — a real, deliberate tradeoff for the cross-asset
structure a single-asset env cannot express at all.

## Result 1: the same "smoother ratio, worse absolute outcome" pattern as CVaR

Single validation window (2024-03-15 → 2024-05-31, 100k timesteps, seed
42), joint policy vs. the sum of the 5 existing independent policies
(`models/ppo_<asset>/model.zip`) replayed on the same date-aligned window:

| | Cum. Return | Sharpe | Sortino | Max DD |
|---|---:|---:|---:|---:|
| Joint portfolio policy | −0.211 | **1.380** | **3.091** | −0.663 |
| Sum of 5 independent policies | **−0.004** | 0.288 | 0.324 | **−0.366** |

Same shape as `docs/risk_sensitivity_ablation.md`'s CVaR finding: **the
joint policy is dramatically better on risk-adjusted ratios (Sharpe
+380%, Sortino +854%) but worse on cumulative return and max drawdown.**
This is now a pattern that has shown up twice independently (a
CVaR-reshaped objective, and now a jointly-trained multi-asset policy) —
suggestive that "smoother, more consistent exposure" is something PPO on
this environment's reward function reaches for whenever it's given a
richer way to express it, not a coincidence specific to either mechanism.
Consistent with the rest of this project's methodology: **this single
window result is not walk-forward validated** — given the date-intersection
cost already shrinks the usable history to 270 days, a full walk-forward
retrofit (which would further subdivide that into rolling 6mo/1mo windows)
was judged not to leave enough data per roll to be meaningful, and was not
attempted. Treat this as a real, reproducible single-window effect, not a
walk-forward-confirmed one.

## Result 2: the joint policy learned real cross-asset structure, unprompted

The interesting question Upgrade 5 set out to answer: does a jointly
trained policy learn genuine cross-asset behavior, or does it just learn 5
independent sub-policies that happen to share a network? Checked directly
by correlating each asset's action time series over the validation window:

|       | DOGE | GME | AMC | PEPE | SHIB |
|---|---:|---:|---:|---:|---:|
| DOGE | 1.000 | 0.258 | 0.264 | **0.700** | **0.665** |
| GME | 0.258 | 1.000 | 0.397 | −0.212 | −0.011 |
| AMC | 0.264 | 0.397 | 1.000 | 0.106 | 0.166 |
| PEPE | **0.700** | −0.212 | 0.106 | 1.000 | **0.643** |
| SHIB | **0.665** | −0.011 | 0.166 | **0.643** | 1.000 |

**DOGE, PEPE, and SHIB — the three small-cap crypto assets — all
correlate 0.64–0.70 with each other**, distinctly higher than any
crypto/meme-stock pair (GME-PEPE is actually *negative*, −0.212). The
policy was never told which assets are "crypto" vs. "meme stock" — this
clustering emerged purely from learning on the joint observation, and it
recovers exactly the asset-class structure this project's own basket
design (`configs/assets.yaml`) already assumes. This is the clearest
piece of evidence in this entire upgrade path that joint training
achieves something a single-asset architecture structurally cannot: the
policy is moving correlated assets together, not treating each one in
isolation.

No collapse: every asset's action std is 0.68–0.77 (per-asset breakdown in
git history of this analysis), with no `frac_near_max_*` anywhere near a
degenerate 0.90+.

## Honest scope note

Single seed (42), single window, one training run. Given the strength and
specificity of the cross-asset correlation finding, a natural next check
would be whether it's seed-robust (does a different seed also discover the
crypto cluster, or land on a different but equally coherent structure?) —
not attempted here given time constraints, and a real open question, not
a claimed-and-verified one.
