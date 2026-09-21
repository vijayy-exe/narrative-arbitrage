# Narrative Arbitrage — Development Plan & Prompt Pack

## 1. Project Context

**One-line pitch:** An RL agent that trades narrative-driven assets (meme stocks, small-cap crypto, NFTs) using only the *velocity and shape* of public attention — Wikipedia edits, Reddit activity, Google Trends — never price-technical indicators.

**Core hypothesis:** For narrative-driven assets, price is a *lagging* indicator of collective attention. The crowd talks, argues, and searches before it buys. If true, an agent watching attention *derivatives* (speed, acceleration, divergence across platforms) should be able to learn a profitable, risk-aware trading policy without ever seeing a price chart as input.

**Why RL and not supervised learning:** This is a sequential decision problem with path-dependent costs (transaction fees, slippage, position sizing over time), not a one-shot prediction task. Supervised learning appears only as a *baseline* to beat, not as the core method.

**State (MDP):**
- First- and second-order derivatives of multi-source attention signals (Wikipedia pageview/edit velocity & acceleration, Reddit comment acceleration, Google Trends slope, cross-platform sentiment divergence, edit-war ratio, novelty score)
- Minimal portfolio context (current position, cash, unrealized P&L)

**Action:** Continuous position-sizing decision (Box space, e.g. -1 to +1 = short to long)

**Reward:** Risk-adjusted P&L net of transaction costs/slippage — not raw return

**Algorithm:** PPO first (simpler to debug), then RecurrentPPO (attention dynamics are sequential, so memory should help)

**Evaluation:** Walk-forward backtesting (rolling train/validate/test, periodic retraining) against two baselines — naive momentum and supervised learning on the same attention features. Metrics: Sharpe, Sortino, max drawdown, total return. Plus feature attribution (SHAP/permutation importance) to identify which attention signals the policy relies on per asset class.

**What makes this a strong project, not a template:**
- The feature engineering (edit-war ratios, cross-platform divergence, novelty scores) is genuinely novel data work, not copy-pasted from a GitHub RL-trading tutorial.
- Look-ahead-bias discipline and honest baselines are enforced throughout — this is the part most "RL trading bot" projects fake.
- A negative result (attention doesn't lead price) is still a publishable-adjacent finding about how information propagates across platforms — the project doesn't live or die on profitability alone.

**Time budget guide:** Data pipeline (Phase 1) ~35%, RL environment + training (Phases 3–4) ~35%, everything else ~30%. Most RL trading projects fail from rushing the data and over-engineering the algorithm — resist that here.

**Real data, not simulated:** Every source in Phase 1 (Wikipedia, Reddit, Google Trends, price) pulls from live public APIs on real historical dates — nothing here is synthetic or mocked for the actual pipeline. Phase 7 goes further and adds a genuinely live loop: the trained agent watching today's attention signals update in real time and making a live paper-trading decision, tracked in a locally simulated portfolio (no paid broker subscription needed — see Phase 7 for the free data sources used). This is what turns the project from "a backtest on a spreadsheet" into something you can actually demo running live in front of a panel — which matters a lot for a final-year evaluation.

---

## 2. How to Use This Document

Each phase below has:
- **Goal** — what "done" looks like
- **Key decisions** — things to lock in before coding
- **Dev prompt** — copy-paste this into a coding assistant (Claude Code, Cursor, etc.) to scaffold that phase. Each prompt assumes the previous phases' outputs already exist in the repo.

Work through phases in order. Don't start Phase 3 (RL env) until Phase 2 (baselines) are running — they catch data bugs far more cheaply than a misbehaving RL agent will.

---

## 3. Phase 0 — Scope & Environment Setup

**Goal:** Repo skeleton, dependencies installed, MDP formally specified in writing before any code touches it.

**Key decisions to make first:**
- Final asset basket: 2–3 meme stocks, 3–4 small-cap crypto, 2–3 NFT collections (pick ones with enough historical data — at least 12–18 months)
- Episode structure: one episode per asset per time window, or multi-asset portfolio in one episode?
- Reward function exact formula (write it as an equation, not just "risk-adjusted P&L")

**Dev prompt — Phase 0:**
```
Set up a Python project called "narrative-arbitrage" for a reinforcement learning
trading research project. Requirements:

- Use Poetry or a requirements.txt with: gymnasium, stable-baselines3[extra],
  pandas, numpy, pytrends, praw, requests, yfinance, ccxt (for crypto),
  scikit-learn, xgboost, shap, matplotlib, pytest
- Repo structure:
  narrative-arbitrage/
    data/raw/, data/processed/
    src/data_pipeline/     (one module per source: wikipedia.py, reddit.py,
                             trends.py, price.py, features.py)
    src/env/                (gym environment)
    src/baselines/          (momentum.py, supervised.py)
    src/agents/             (training scripts)
    src/eval/               (backtesting, attribution)
    notebooks/               (exploration only, no pipeline logic here)
    tests/
    configs/                (YAML configs for assets, date ranges, hyperparams)
- Add a configs/assets.yaml listing placeholder tickers/symbols for meme stocks,
  small-cap crypto, and NFT collections
- Add a docs/mdp_spec.md file where I will write out the state space, action
  space, and reward formula in plain English + math notation before any
  environment code is written
- Set up pytest with a placeholder test in each module folder
- Do NOT write any data-fetching or RL logic yet — this step is scaffolding only
```

---

## 4. Phase 1 — Data Pipeline

**Goal:** Clean, aligned, leak-free time series of attention derivatives for every asset in the basket, with price data kept strictly separate (used only for reward computation, never as a feature).

**Key decisions:**
- Timestamp alignment strategy (daily bars vs. hourly — daily is more forgiving to start)
- How to handle missing data per source (forward-fill vs. drop vs. interpolate — document the choice, it affects leakage risk)
- Definition of each derived feature: edit-war ratio, novelty score, cross-platform sentiment divergence

**Dev prompt — Phase 1a (raw ingestion):**
```
In src/data_pipeline/, implement data fetchers for each attention source, one
module per source. For each asset in configs/assets.yaml, over the date range
in configs/date_range.yaml:

1. wikipedia.py — pull daily pageviews (Wikimedia REST API) and edit history
   (MediaWiki API) for the asset's Wikipedia page. Output: date, pageviews,
   edit_count, unique_editors, revert_count.
2. reddit.py — pull daily comment counts and top-level post counts from
   relevant subreddits (use PRAW with read-only auth). Output: date,
   comment_count, post_count, avg_sentiment (use VADER for now).
3. trends.py — pull daily/weekly Google Trends interest-over-time via pytrends
   for the asset's search term. Output: date, search_interest.
4. price.py — pull daily OHLCV price data (yfinance for stocks, ccxt for
   crypto). This is kept in data/raw/price/ and is NEVER imported by the
   feature engineering module — only by the reward/backtest code later.

Each fetcher should cache raw responses to data/raw/<source>/<asset>.csv and
be idempotent (skip re-fetching if the file exists and covers the date range).
Write a pytest test per fetcher using a small mocked date range.
```

**Dev prompt — Phase 1b (feature engineering — the core novel work):**
```
In src/data_pipeline/features.py, build the attention-derivative feature set
from the raw ingested data in data/raw/. For each asset and each day t:

- First derivative (velocity) and second derivative (acceleration) of:
  pageviews, edit_count, comment_count, search_interest
- Edit-war ratio: revert_count / edit_count (rolling window, handle
  divide-by-zero)
- Novelty score: how much today's discussion content differs from a rolling
  window of prior discussion (e.g. cosine distance between TF-IDF vectors of
  daily Reddit comment text vs. the trailing 7-day window)
- Cross-platform sentiment divergence: difference between Reddit sentiment
  and any other available sentiment signal, and divergence in *timing* (does
  Wikipedia edit velocity spike before or after Reddit comment velocity?)

CRITICAL: every feature at time t must only use data available up to and
including time t. Add an explicit assertion/test that checks no feature
function looks at future rows. Output one processed CSV per asset to
data/processed/<asset>_features.csv with columns clearly named and documented
in a data dictionary (docs/feature_dictionary.md).

Also produce a quick sanity-check notebook that plots attention-derivative
features against price for 2-3 well-known cases (e.g. a known meme stock
squeeze) so I can visually confirm the pipeline captures the expected
pre-price attention spike.
```

---

## 5. Phase 2 — Baselines (Before Touching RL)

**Goal:** A momentum baseline and a supervised-learning baseline running end-to-end on the same processed features, giving a floor to beat and catching leakage bugs while they're still cheap to fix.

**Dev prompt — Phase 2:**
```
In src/baselines/, implement two baseline strategies using the processed
features in data/processed/:

1. momentum.py — a naive price-momentum strategy (e.g. go long if trailing
   N-day return is positive, flat/short otherwise) as a sanity-check
   benchmark. This uses price only, not attention features.

2. supervised.py — an XGBoost (or logistic regression) classifier/regressor
   that predicts next-period return direction/magnitude from the attention-
   derivative features built in Phase 1b (NOT from price). Use a proper
   time-series train/test split (no shuffling, no random k-fold) and report
   accuracy/AUC plus a simple backtest of a strategy that acts on its
   predictions.

For both, output a small report (markdown or notebook) with:
- Cumulative return chart
- Sharpe, Sortino, max drawdown
- Explicit statement of the train/test date boundaries used, to make
  look-ahead-bias auditing easy later

Also write a tests/test_no_leakage.py that fits the supervised baseline on
a truncated dataset and asserts its predictions don't change when future
rows (beyond the truncation point) are added to the input — this is a
regression test for feature leakage that Phase 1b should already satisfy,
but this test guards against future changes.
```

---

## 6. Phase 3 — RL Environment

**Goal:** A working `gymnasium.Env` implementing the MDP specified in `docs/mdp_spec.md`, tested with random actions before any learning algorithm touches it.

**Dev prompt — Phase 3:**
```
In src/env/, implement a custom gymnasium.Env called NarrativeTradingEnv
following the spec in docs/mdp_spec.md:

- Observation space: Box containing the attention-derivative features from
  data/processed/<asset>_features.csv for the current timestep, plus
  portfolio context (current position size, cash, unrealized P&L) —
  normalize/scale all inputs.
- Action space: Box(-1, 1) representing target position size (short to long).
- Step logic: apply the action, compute transaction cost/slippage based on
  the change in position, advance to the next timestep, compute reward as
  [use the exact formula from docs/mdp_spec.md — do not invent a different
  one], return next observation.
- Episode: one episode = one asset's full date range (start with single-asset
  episodes; leave a TODO for multi-asset portfolio episodes later).
- Include render() that prints/logs position, cash, and reward for debugging.

Write tests/test_env.py that:
- Steps through a full episode with random actions and asserts no NaNs/infs
  appear in observations or rewards
- Asserts a "do nothing" (always hold position 0) policy yields ~zero reward
  minus any fixed costs
- Asserts reward sign is correct for an obviously profitable synthetic
  scenario (construct fake data where going long is trivially profitable)

Do not start training an agent in this phase — this is environment
correctness only.
```

---

## 7. Phase 4 — Agent Training

**Goal:** A PPO agent that learns something non-trivial on a single asset, then RecurrentPPO across the full basket.

**Dev prompt — Phase 4a (single-asset PPO):**
```
In src/agents/, using stable-baselines3, write a training script that:
- Trains a PPO agent on NarrativeTradingEnv for a single asset (pick the one
  with the longest, cleanest history from Phase 1)
- Logs training curves (episode reward, Sharpe of the resulting policy on a
  held-out validation window) via TensorBoard
- Saves checkpoints to models/ppo_<asset>/
- After training, runs the policy on the validation window and compares its
  Sharpe/Sortino/max drawdown against the Phase 2 baselines for the same
  asset and window

Also add a script src/eval/reward_hacking_check.py that inspects the trained
policy's action distribution over the validation window — flag if the agent
has collapsed to always-hold-cash or always-max-long, which would indicate
reward hacking rather than genuine learning.
```

**Dev prompt — Phase 4b (multi-asset RecurrentPPO):**
```
Extend the training setup from Phase 4a to use RecurrentPPO (sb3-contrib)
and train across the full asset basket from configs/assets.yaml. Options to
implement (pick based on what's simplest given the env from Phase 3):
(a) train one shared policy across assets by cycling episodes through the
    basket, or
(b) train separate policies per asset and compare.
Log per-asset validation performance the same way as Phase 4a. Compare
RecurrentPPO against the plain PPO results from 4a to see whether the
sequential memory actually helps — record this comparison explicitly, it's
relevant to the write-up either way.
```

---

## 8. Phase 5 — Walk-Forward Backtesting

**Goal:** Rigorous, leakage-free performance evaluation across rolling windows, not a single train/test split.

**Dev prompt — Phase 5:**
```
In src/eval/, implement a walk-forward backtesting harness:
- Split each asset's date range into rolling windows (e.g. train on 6 months,
  validate on 1 month, roll forward by 1 month, retrain)
- Retrain (or fine-tune) the RL agent and both baselines at each roll
- Aggregate out-of-sample performance across all windows: Sharpe, Sortino,
  max drawdown, total return, plus win rate and average trade duration
- Produce a comparison table/chart: RL agent vs. momentum baseline vs.
  supervised baseline, per asset and averaged across the basket
- Write an explicit look-ahead-bias audit script that re-checks every
  feature and every train/retrain boundary respects the "no future data"
  rule from Phase 1b, now applied at the walk-forward level

Output a docs/backtest_report.md summarizing results with the comparison
table and charts embedded.
```

---

## 9. Phase 6 — Feature Attribution & Write-Up

**Goal:** Interpretability results (which attention signals matter, per asset class) and a project write-up suitable for a portfolio/application.

**Dev prompt — Phase 6:**
```
In src/eval/attribution.py, run feature attribution on the trained RL
policy's action outputs with respect to the input features (SHAP or
permutation importance — adapt for a continuous-action policy network).
Break results down per asset class (meme stock vs. crypto vs. NFT) and
produce:
- A bar chart of average feature importance per asset class
- Specific cross-platform lead-lag findings, e.g. does Wikipedia edit
  velocity predictably precede Reddit comment velocity for one asset class
  but not another — check this explicitly with a cross-correlation analysis
  between the raw (pre-derivative) signals

Then draft docs/final_report.md covering: problem statement, method,
data pipeline summary, backtest results vs. baselines, attribution findings,
and an honest discussion of where the hypothesis held vs. failed. Keep this
report ready to reuse as a portfolio writeup or SOP supporting evidence.
```

---

## 10. Phase 7 — Real-Time Deployment & Live Web App (Final-Year Centerpiece)

**Goal:** The trained agent running on *today's* data, paper-trading live inside a proper web app with a simulated account — no paid broker subscription required, entirely free data sources.

**Why this phase matters for a final-year project specifically:** Evaluators see dozens of "trained a model, here's a backtest chart" projects. A live-running system that pulls today's Wikipedia/Reddit/Trends data, feeds it through your trained policy, and shows a decision update in a real web app is a different tier of demo — it proves the pipeline actually works end-to-end, not just on a curated historical CSV.

**Free real-time data sources (no subscriptions):**
- **Stocks (NSE/BSE):** `yfinance` — free, no API key, ~15-20 min delayed price data. Delay is irrelevant here since your features update daily, not tick-by-tick.
- **Crypto:** Binance public REST/WebSocket API — genuinely free, real-time, no key needed for market data. For paper-trading order simulation, Binance Testnet gives you a free sandbox account with fake funds and a real order-matching engine.
- **NFTs:** OpenSea's public API (free tier) for floor price / volume data, if you keep an NFT asset in the basket — otherwise fine to drop NFTs to backtest-only and lean the live demo on stocks + crypto.
- **Attention signals:** Wikipedia, Reddit (PRAW free tier), Google Trends (pytrends) — all already free, same as Phase 1.
- **Your "account":** rather than connecting a real broker, the app maintains its own simulated portfolio (cash, positions, P&L) in a local database, updated from the free live price feeds above. This is safer for a project anyway — no real money at risk on an experimental model — and avoids any paid API dependency entirely.

**Web app architecture (instead of Streamlit):**
- **Backend:** FastAPI (Python) — reuses your existing pipeline/model code directly, exposes REST endpoints for live features, current agent decision, portfolio state, and backtest results
- **Frontend:** Next.js + a charting library (Recharts or Chart.js) — a real web app you can deploy (Vercel for frontend, Render/Railway free tier for backend) and share a live link, not just a local script
- **Scheduler:** a background job (APScheduler inside the FastAPI process, or a simple cron) that refreshes live data and re-runs inference every N hours, writes to the local DB, which the frontend polls

**Dev prompt — Phase 7:**
```
Build a real-time web app on top of the trained agent from Phase 4/5, using
only free data sources (no paid broker API):

1. src/realtime/scheduler.py — a scheduled job (APScheduler) that runs every
   N hours: fetches today's attention data (Wikipedia/Reddit/Trends, same
   fetchers as Phase 1) and today's price via yfinance (stocks) and Binance
   public API (crypto). Recomputes attention-derivative features using the
   exact same functions from src/data_pipeline/features.py. Appends to a
   local SQLite database (data/live/live.db).

2. src/realtime/infer.py — loads the latest trained checkpoint, builds the
   current observation from the live feature store, gets the agent's action,
   and updates a simulated portfolio table in the same SQLite DB (cash,
   positions, mark-to-market P&L using the latest free price). Logs every
   decision with a timestamp. No real orders are ever placed.

3. backend/ — a FastAPI app exposing:
   GET /api/features/{asset}      -> latest attention-derivative values
   GET /api/decision/{asset}      -> agent's current action + reasoning
   GET /api/portfolio             -> simulated portfolio state, live P&L
   GET /api/backtest/{asset}      -> Phase 5 walk-forward results, for
                                      side-by-side comparison with live

4. frontend/ — a Next.js app with:
   - A dashboard page per asset: live attention-derivative charts, agent's
     current position, live vs backtested performance comparison
   - A portfolio page: simulated cash/positions/P&L, decision log/history
   - Simple, clean layout — a couple of chart components and a table is
     enough, don't over-build the UI

Rate-limit and cache all API calls appropriately (Reddit/Google Trends have
strict free-tier limits). Keep the whole loop resilient to a single source
failing (e.g. if Google Trends rate-limits you, still update the other
features and log the gap rather than crashing the scheduler).
```

**Fallback if live APIs are unreliable during the actual demo:** Keep a `--replay` mode that replays a recorded day from your walk-forward test set at accelerated speed through the same web app, so the demo still looks live even if a free API rate-limits you mid-presentation. Mention in your report that this is a replay safeguard, not a substitute for the real live loop (which should still run and be shown working beforehand, e.g. in a screen recording).

---

## 11. Final-Year Project Fit — Scope, Timeline, and Fallbacks

**Realistic semester mapping (assuming ~14–16 weeks):**

| Weeks | Phase(s) |
|---|---|
| 1–2 | Phase 0 — setup, MDP spec, asset basket finalized |
| 3–6 | Phase 1 — data pipeline + feature engineering (biggest chunk, budget extra buffer) |
| 6–7 | Phase 2 — baselines running, sanity-checked |
| 7–9 | Phase 3 — RL environment built and tested |
| 9–11 | Phase 4 — PPO then RecurrentPPO training |
| 11–12 | Phase 5 — walk-forward backtest |
| 12–13 | Phase 6 — attribution + report |
| 13–14 | Phase 7 — live demo loop + dashboard |
| 14–16 | Buffer, polish, rehearse the demo |

**If you're running short on time, cut in this order (least damaging to most):**
1. Trim the asset basket (6 assets instead of 10 is fine — depth over breadth)
2. Drop RecurrentPPO, keep plain PPO (still a complete, defensible project)
3. Reduce walk-forward windows (fewer rolls, same methodology, still honest)
4. If NFT data sourcing gets flaky, drop NFTs from the live loop and keep them backtest-only — stocks + crypto alone is a complete live demo
5. Never cut: the "no future data" leakage checks, the baselines, or Phase 7's live web app — these are what separate this from a generic tutorial project and are what evaluators will actually probe on.

**What makes this evaluator-proof in a viva:**
- You can answer "how do you know it's not just overfit?" → walk-forward + leakage tests (Phase 1b, Phase 5)
- You can answer "why RL and not just a classifier?" → sequential decisions with transaction costs, not one-shot prediction (already in your problem statement)
- You can show it running *live*, not just a static notebook

---

## 12. Order of Operations Checklist

- [ ] Phase 0 — repo scaffolded, MDP spec written
- [ ] Phase 1a — raw data fetchers working for all sources
- [ ] Phase 1b — attention-derivative features built, leakage-checked
- [ ] Phase 2 — momentum + supervised baselines running, reports generated
- [ ] Phase 3 — gym env implemented and tested with random actions
- [ ] Phase 4a — PPO trained on single asset, beats or approaches baselines
- [ ] Phase 4b — RecurrentPPO across full basket
- [ ] Phase 5 — walk-forward backtest complete, comparison table produced
- [ ] Phase 6 — attribution analysis + final report drafted
- [ ] Phase 7 — live real-time loop + web app working (free data sources), demo rehearsed
