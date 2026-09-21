# Architecture Ablation — MLP vs LSTM (RecurrentPPO) vs Transformer

> **⚠ UPDATE (2026-09-20): this document's headline finding does NOT
> survive walk-forward re-validation, except narrowly for GME.**
> Walk-forward basket-average Sharpe: LSTM 0.451, Transformer 0.512 — both
> WORSE than the original plain MLP's 0.712, reversing the single-window
> claim below that LSTM wins on every basket-average metric. The one real,
> walk-forward-confirmed win is **LSTM specifically on GME** (−1.00 → +1.38)
> — a genuine, large improvement on the basket's worst asset. Everywhere
> else, including AMC (this document's other headline single-window win,
> Transformer +0.351), the walk-forward number is flat-to-negative. See
> `docs/walk_forward_upgrade_validation.md` for the full re-validation and
> per-asset detail. Everything below is preserved as an accurate record of
> the single-window result, but should not be read as "LSTM/Transformer
> generally beat the MLP" — walk-forward says that's true for GME only.

Upgrades 2+3 of `narrative-arbitrage-implementation-upgrade.md`, run together
as that document's own "Suggested Stopping Points" section groups them: a
controlled 3-way comparison of *how the policy represents the state*,
holding data/reward/seed fixed. This directly answers the original project's
cut hypothesis ("does sequential memory help?", Phase 4b, skipped —
`docs/CURRENT_STATUS.md`) and extends it with a genuine attention-mechanism
encoder, matching the project's own thematic premise ("an attention
mechanism trading on attention" — `narrative-arbitrage-context-upgrade.md`
§2.2).

**Method:** 100,000 timesteps, seed 42, identical train/validation split per
asset (`src/agents/common.get_train_val_split`) — same protocol as every
other single-window number in this project. Code: `src/agents/
architecture_ablation.py`.

- **MLP** — the original Upgrade 0/1 "none" baseline checkpoints
  (`models/ppo_<asset>/`), reused as-is, NOT retrained, evaluated on the
  identical validation window as the other two arms.
- **LSTM** — `sb3-contrib`'s `RecurrentPPO` (`MlpLstmPolicy`), single-
  timestep observation (`lookback_window=1`), memory carried via the LSTM's
  own persistent hidden state across the full episode.
- **Transformer** — `src/agents/attention_encoder.py`'s
  `TemporalAttentionExtractor` (2-layer, 4-head, `d_model=32`) plugged into
  standard PPO via `policy_kwargs`, fed a 10-day lookback window
  (`src/env/trading_env.py`'s new optional `lookback_window` parameter,
  default 1 = unchanged for every other checkpoint in this repo).

**Design decision — why LSTM does NOT also get the lookback window:** an
LSTM's mechanism already IS cross-episode memory; stacking a redundant
lookback window in front of it would process overlapping history through
two different memory mechanisms at once, muddying what's actually being
tested. LSTM gets the natural single-timestep stream; only the (stateless)
Transformer needs an explicit window. See `architecture_ablation.py`'s
module docstring for the fuller reasoning — this is a deliberate deviation
from the implementation doc's literal wording ("same lookback-window
observation from Upgrade 2"), documented rather than silently done.

## Results: all 5 assets × 3 architectures, single validation window

| Asset | Arch | Sharpe | Sortino | Max DD | Cum. Return |
|---|---|---:|---:|---:|---:|
| AMC | MLP (baseline) | −2.030 | −2.456 | −0.911 | −0.908 |
| AMC | LSTM | −1.048 | −0.858 | −1.622 | −1.597 |
| AMC | **Transformer** | **+0.351** | **+0.442** | −0.441 | −0.068 |
| GME | MLP (baseline) | −3.816 | −3.437 | −1.191 | −0.889 |
| GME | **LSTM** | **−0.035** | **−0.070** | −0.899 | −0.751 |
| GME | Transformer | −1.662 | −1.227 | −1.362 | −0.965 |
| DOGE | **MLP (baseline)** | **+3.953** | **+5.815** | **−0.132** | **+1.429** |
| DOGE | LSTM | +2.139 | +4.391 | −0.256 | +0.681 |
| DOGE | Transformer | +1.654 | +2.988 | −0.290 | +0.472 |
| PEPE | MLP (baseline) | +0.373 | +0.631 | −0.581 | −0.075 |
| PEPE | LSTM | +1.825 | +4.242 | −0.282 | +0.604 |
| PEPE | **Transformer** | **+2.818** | **+5.339** | **−0.227** | **+0.871** |
| SHIB | MLP (baseline) | +0.885 | +0.803 | −0.558 | +0.093 |
| SHIB | **LSTM** | **+2.923** | **+10.671** | **−0.174** | **+1.334** |
| SHIB | Transformer | +1.202 | +3.692 | −1.201 | −0.643 |
| **Basket avg** | MLP (baseline) | −0.127 | +0.271 | −0.675 | −0.070 |
| **Basket avg** | **LSTM** | **+1.161** | **+3.675** | **−0.647** | **+0.054** |
| **Basket avg** | Transformer | +0.873 | +2.247 | −0.704 | −0.067 |

Bold marks the best architecture per asset/metric column, and the
basket-average winner. Collapse check (`classify_action_distribution`) run
on every one of the 10 new checkpoints — **none collapsed** (all `std_action`
∈ [0.10, 0.98], well clear of the 0.05 threshold; no `frac_near_max_*`
anywhere near the 0.90 trigger).

## Headline finding: sequential memory helps, decisively, on average

**LSTM (RecurrentPPO) wins on every basket-average metric** — best Sharpe
(1.161 vs MLP's −0.127), best Sortino (3.675 vs 0.271), best (least
negative) max drawdown (−0.647 vs −0.675), and the **only architecture with
a positive average cumulative return** (+0.054, vs MLP's −0.070 and
Transformer's −0.067). This is a real answer to the original project's
"does RecurrentPPO help over plain PPO?" question that Phase 4b's scope cut
left untested — on this evidence, **yes, more than either the flat MLP or
the Transformer encoder, and more than Upgrade 1's CVaR objective**
(basket-average Sharpe 0.456, Sortino 0.902 — both below LSTM's numbers
here; see `docs/risk_sensitivity_ablation.md`).

Compared against Upgrade 1's CVaR result specifically: LSTM beats CVaR on
every basket-average metric, including the two CVaR struggled with most
(cumulative return and max drawdown) — architecture (giving the policy
memory) generalized better here than reshaping the objective (making the
critic pessimistic) did, at least on this single-window evidence.

## But no single architecture wins per-asset — a genuinely mixed picture

- **DOGE**: the original flat MLP wins on all 4 metrics (Sharpe 3.953,
  Sortino 5.815, max DD −0.132, cum. return +1.429, all the best of the
  three) — more capacity/memory did not help the one asset that was already
  working well; LSTM is second-best across the board here, not first on
  anything. DOGE is the one asset where "simpler is better" holds.
- **AMC**: Transformer is the clear winner (+0.351 Sharpe, the only
  positive number this asset has produced under ANY objective/architecture
  tried across Upgrades 1-3) — LSTM actually makes AMC's cumulative return
  *worse* than the MLP baseline (−1.597 vs −0.908).
- **GME**: LSTM is the clear winner (−0.035 Sharpe — not positive, but by
  far the closest to breakeven GME has gotten) — Transformer helps GME
  less than LSTM does, though both beat the MLP baseline's −3.816.
- **PEPE and SHIB**: both new architectures beat the MLP baseline by a wide
  margin, with Transformer best for PEPE and LSTM best for SHIB.

**No architecture is uniformly correct to deploy.** If forced to pick one
architecture for the whole basket, LSTM's basket-average dominance makes it
the defensible default — but AMC specifically would be better served by the
Transformer, and DOGE specifically would be better served by keeping the
original MLP. A per-asset architecture selection (not attempted here, flagged
as a further extension) would likely beat any single basket-wide choice.

## Scope note — single window, not walk-forward

Same caveat as `docs/risk_sensitivity_ablation.md`: all 15 runs above are
Phase-4a-style single-window comparisons (100k timesteps, one
train/validate split, seed 42), not Phase 5's walk-forward harness. Given
`docs/CURRENT_STATUS.md` already documents single-window and walk-forward
numbers disagreeing substantially even for the ORIGINAL MLP-only PPO runs
(DOGE: 3.95 single-window vs 1.71 walk-forward average), the same
uncertainty applies here, likely more so given LSTM/Transformer have more
parameters to overfit a short window with. **Walk-forward validation of
both new architectures is the top follow-up** before trusting the "LSTM
wins on average" headline finding beyond "true for this specific window."

## What's in `models_arch/`

`models_arch/lstm_<asset>/model.zip` and `models_arch/transformer_<asset>/
model.zip` for all 5 assets — kept alongside `models/` (original) and
`models_cvar/` (Upgrade 1) rather than overwriting either, so all three
objective/architecture combinations remain available for Upgrade 4's
multi-seed reporting or a future walk-forward rerun.
