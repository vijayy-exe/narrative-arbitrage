# Attribution Extended to All 5 Assets

`docs/final_report.md` §5a and `docs/attribution_report.md` both flagged
extending permutation-importance attribution beyond DOGE as "the single
most actionable follow-up" — deferred pending AMC/GME clearing a
reward-hacking audit (`docs/reward_hacking_findings.md`, done in Upgrade 0)
before trusting their attribution output. That audit passed; this
document closes the gap using `src/eval/attribution.py --asset <X>`
(code already supported any asset via `--asset`, it had just only ever
been invoked for DOGE) against each asset's ORIGINAL `models/ppo_<asset>/
model.zip` checkpoint (the Upgrade 0/1 "none" baseline, not any of the
CVaR/LSTM/Transformer checkpoints from the upgrade path) — same method as
DOGE's original analysis: 5-repeat permutation on each feature, actually
replayed through the environment for a real counterfactual Sharpe, not
just a raw action-shift number. Full per-asset tables: `docs/
attribution_report_{AMC,GME,PEPE,SHIB}.md`; DOGE's original is `docs/
attribution_report.md`.

## Top feature per asset

| Asset | #1 feature | mean \|action shift\| | #2 feature | #3 feature |
|---|---|---:|---|---|
| DOGE | `search_interest_velocity` | 0.326 | `search_interest` | `search_interest_acceleration` |
| PEPE | `search_interest_velocity` | 0.277 | `pageviews` | `hn_story_count_acceleration` |
| SHIB | `search_interest` (level) | 0.712 | `pageviews` (level) | `pageviews_velocity` |
| AMC | `edit_count_acceleration` | 0.336 | `pageviews_velocity` | `search_interest` |
| GME | `pageviews_acceleration` | 0.458 | `pageviews_velocity` | `hn_comment_count_velocity` |

## A real, class-correlated pattern — with one clean exception

**Small-cap crypto (DOGE, PEPE) relies most on Google Trends VELOCITY** —
exactly the "attention accelerating, not just attention level" signal this
project's core hypothesis is built around (`docs/final_report.md` §1). Both
assets' #1 feature is `search_interest_velocity`, corroborating the raw
return-correlation finding already documented in `src/data_pipeline/
features.py`'s `recompute_derivatives_on_index` docstring (0.23 for DOGE,
0.38 for SHIB — though note that correlation number is about SHIB, while
this permutation-importance finding is about DOGE/PEPE; the two methods
agree where they overlap on DOGE, and diverge for SHIB, which is exactly
this section's clean exception below).

**Meme stocks (AMC, GME) rely most on Wikipedia ACCELERATION** —
`edit_count_acceleration` for AMC, `pageviews_acceleration` for GME —
not Google Trends at all. This is a genuinely different attention channel
mattering for a different asset class, consistent with `docs/
final_report.md` §5b's independent finding (from raw cross-correlation,
no trained policy needed) that meme-stock Wikipedia/Trends correlation is
much weaker than crypto's (AMC 0.16, GME 0.28 vs. crypto's ≈0.49-0.88) —
if the two platforms move less together for meme stocks, it makes sense
the POLICY learns to lean on whichever one actually carries the signal
(Wikipedia) rather than the one this project's crypto-derived intuition
would predict (Trends).

**SHIB is the clean exception to the "velocity matters most" thesis.**
Its #1 feature is `search_interest` at LEVEL (0.712, more than double any
other feature/asset combination found across the whole basket), not
`search_interest_velocity` (which ranks 6th for SHIB, mean shift only
0.107). This means the core "derivatives matter more than levels"
hypothesis this entire project is built on does NOT hold for every asset
— SHIB's policy learned that raw attention level, not its rate of change,
is what matters. This is a genuine, previously-invisible finding (the
DOGE-only analysis had no way to surface it) and a real limitation on how
broadly this project's headline framing ("price is a lagging indicator of
attention derivatives") actually generalizes across its own basket.

## Caveat carried over from the single-asset analysis, still applies

These are permutation-importance results on each asset's ORIGINAL,
single-window (Phase 4a) checkpoint — not walk-forward validated, and
(per `docs/multiseed_report.md`'s separate finding) not multi-seeded
either. Given how much the walk-forward re-validation
(`docs/walk_forward_upgrade_validation.md`) changed the *architecture*
story, it's plausible feature-importance rankings could also shift under
a walk-forward-averaged or multi-seeded analysis — that combination
(attribution × walk-forward, or attribution × multi-seed) was not
attempted in this pass and is a further, still-open extension.
