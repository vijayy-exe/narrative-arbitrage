# Narrative Arbitrage

**Does attention lead price?** An RL agent that trades meme stocks and
small-cap crypto using *only* the velocity and shape of public attention —
Wikipedia edit acceleration, Google Trends search-interest slope, Hacker
News discussion volume — and never once looks at a price chart as input.

This is not a "trained a trading bot, line went up" project. It's a record
of an entire RL upgrade path — a risk-sensitive objective, a Transformer
encoder, RecurrentPPO, a joint multi-asset policy, an ensemble — being
built, looking like clear wins on a single validation window, and then
getting re-validated under a proper walk-forward protocol. **Most of it
didn't survive.** That reversal, and what it took to catch it, is the
actual headline result. Read the full story in the live dashboard's
[**Research**](#running-it-locally) page, or [`docs/
walk_forward_upgrade_validation.md`](docs/walk_forward_upgrade_validation.md).

## The headline finding

| Method | Walk-forward Sharpe (5-asset basket avg) |
|---|---:|
| Momentum (price-only baseline) | 1.12 |
| **Plain PPO, attention-only** | **0.71** |
| Transformer encoder | 0.51 |
| RecurrentPPO (LSTM) | 0.45 |
| CVaR-objective PPO | 0.11 |
| Supervised (XGBoost) baseline | 0.07 |

Every architecture/objective upgrade built on top of the original PPO
agent **looked like a clear improvement on a single validation window**
and **lost to the original once walk-forward validated**. The one
confirmed exception — LSTM specifically on GME (the basket's worst
performer, −1.00 → +0.89 mean across 3 seeds) — is real but not fully
seed-robust either. See [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)
for the full, continuously-updated state of every claim in this repo.

## What's actually in here

- **A real data pipeline** — Wikipedia, Google Trends, Hacker News, price
  (yfinance/ccxt), and a live-only 4chan `/biz/` signal, all pulled from
  free, no-auth APIs with documented, honest gaps (Reddit is blocked;
  StockTwits and GDELT were evaluated and rejected — see
  [`docs/feature_dictionary.md`](docs/feature_dictionary.md)).
- **A leakage-audited walk-forward backtest harness**
  (`src/eval/walk_forward.py`) — rolling 6-month train / 1-month validate
  windows, retrained from scratch every roll, with an explicit
  no-look-ahead audit that runs before every evaluation.
- **Six RL variants**, all sharing the same environment and reward:
  plain PPO, a genuine quantile-critic CVaR objective
  (`src/agents/distributional_critic.py`), a Transformer temporal-attention
  encoder (`src/agents/attention_encoder.py`), RecurrentPPO, a seed
  ensemble (`src/agents/ensemble.py`), and a joint multi-asset portfolio
  policy (`src/env/multi_asset_env.py`) that discovered real cross-asset
  structure (crypto assets' actions correlate 0.64–0.70 with each other,
  unprompted) with no label ever telling it which assets are crypto.
- **A live paper-trading loop** (`src/realtime/`) — fetches today's
  attention data, runs the trained policy, updates a simulated portfolio.
  Zero real orders, ever.
- **A Next.js + FastAPI dashboard** — live/replay toggle per asset,
  backtest comparison tables, and a full narrative walkthrough of the
  research journey above.
- **~140 tests**, all passing, covering everything from feature-leakage
  audits to end-to-end RL training smoke tests.

## Running it locally

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --port 8000

# Frontend (separate terminal)
cd frontend && npm install
echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000" > .env.local
npm run dev
```

Open `http://localhost:3000` — the dashboard works immediately against
the historical replay data already checked into this repo (no live API
keys needed). Click **Research** for the full walk-forward story with
charts.

To run the live loop for real (fetches today's data from real APIs):
```bash
python -m src.realtime.scheduler --once
python -m src.realtime.infer
```
See [`docs/live_loop_first_run.md`](docs/live_loop_first_run.md) for the
exact cron setup for continuous operation, and what broke the first time
this was ever actually run end-to-end.

## Project structure

```
src/data_pipeline/   Wikipedia / Trends / HN / price / 4chan fetchers + feature engineering
src/env/             the trading environment(s) -- single-asset and joint multi-asset
src/agents/          PPO variants: plain, CVaR, Transformer, LSTM, ensemble, portfolio, BC-pretrain
src/eval/            walk-forward backtest harness, attribution, reward-hacking audit
src/baselines/       momentum and supervised (XGBoost) baselines
src/realtime/        live scheduler + inference loop + SQLite store
backend/             FastAPI app serving live/replay state to the frontend
frontend/            Next.js dashboard
docs/                every finding, honestly reported -- start with CURRENT_STATUS.md
```

## Where to actually start reading

1. [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — the living,
   continuously-corrected source of truth for what's actually true right now.
2. [`docs/walk_forward_upgrade_validation.md`](docs/walk_forward_upgrade_validation.md)
   — the capstone finding: why most of the RL upgrades that looked like
   wins weren't.
3. [`docs/final_report.md`](docs/final_report.md) — the full write-up,
   structured around the original hypothesis, where it held and failed,
   and everything built on top of it.

Everything else in `docs/` is a focused report on one specific piece —
attribution, reward hacking, the hyperparameter sweep, seed variance, the
joint portfolio policy — each one linked from the documents above.

## Tech stack

Python (`gymnasium`, `stable-baselines3` + `sb3-contrib`, `torch`,
`xgboost`, `optuna`), FastAPI, SQLite, Next.js + Tailwind + Recharts.
