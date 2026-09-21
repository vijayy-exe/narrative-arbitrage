"""backend/main.py -- Phase 7 FastAPI app.

Serves the live paper-trading loop's state (data/live/live.db, written by
src/realtime/scheduler.py + src/realtime/infer.py) plus a replay fallback
(backend/replay.py) and the Phase 5 walk-forward backtest numbers
(backend/backtest_report.py), for the Next.js frontend (frontend/).

Run locally:
    uvicorn backend.main:app --reload --port 8000

Every asset/portfolio/decision response carries `"mode"`: "live" or
"replay" (query param `?mode=replay`, default "live") so the two are never
confused. `"method"` on a decision is always exactly "ppo" or
"momentum_baseline" (see src/realtime/db.py's insert_decision assertion) --
DOGE is the only asset with an actual trained PPO checkpoint; every other
asset's decisions are the momentum baseline, clearly labeled as such (see
src/realtime/infer.py's module docstring for the full honesty rationale).
No endpoint here ever places a real order -- everything is a simulated
paper portfolio.
"""
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.realtime import db as livedb
from src.realtime.infer import has_trained_policy
from src.realtime.scheduler import live_assets

from backend import replay as replay_mod
from backend.backtest_report import get_backtest_for_asset

app = FastAPI(title="Narrative Arbitrage -- Live Dashboard API")

# Local demo only (dev plan Phase 7 scope explicitly excludes a real
# deployment) -- wide-open CORS is fine for a localhost:3000 -> localhost:8000
# dev setup and is never exposed publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ModeParam = Literal["live", "replay"]

_LIVE_ASSETS = live_assets()
LIVE_ASSET_SYMBOLS = [a["symbol"] for a in _LIVE_ASSETS]


def _require_known_asset(asset: str):
    if asset not in LIVE_ASSET_SYMBOLS:
        raise HTTPException(
            status_code=404,
            detail=f"{asset!r} is not in the live-loop asset basket ({LIVE_ASSET_SYMBOLS}). "
                   "NFT collections are backtest-only (no live price feed wired) -- see "
                   "src/agents/common.py's available_assets() / docs/feature_dictionary.md.",
        )


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/assets")
def list_assets():
    """Convenience endpoint for the frontend: which assets exist in the
    live loop, and honestly, which method each one actually uses."""
    return {
        "assets": [
            {
                "symbol": a["symbol"],
                "category": a["category"],
                "method": "ppo" if has_trained_policy(a["symbol"]) else "momentum_baseline",
            }
            for a in _LIVE_ASSETS
        ]
    }


@app.get("/api/features/{asset}")
def get_features(asset: str, mode: ModeParam = Query("live")):
    _require_known_asset(asset)

    if mode == "replay":
        return {"mode": "replay", **replay_mod.replay_features(asset)}

    conn = livedb.get_connection()
    try:
        row = livedb.get_latest_feature_row(conn, asset)
        history = livedb.get_feature_history(conn, asset, limit=60)
    finally:
        conn.close()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No live feature data yet for {asset}. Run `python -m "
                   "src.realtime.scheduler --once` first, or pass ?mode=replay.",
        )
    return {"mode": "live", **row, "history": history}


@app.get("/api/decision/{asset}")
def get_decision(asset: str, mode: ModeParam = Query("live")):
    _require_known_asset(asset)

    if mode == "replay":
        return {"mode": "replay", **replay_mod.replay_decision(asset)}

    conn = livedb.get_connection()
    try:
        decision = livedb.get_latest_decision(conn, asset)
        history = livedb.get_decision_history(conn, asset, limit=50)
    finally:
        conn.close()
    if decision is None:
        raise HTTPException(
            status_code=404,
            detail=f"No live decision yet for {asset}. Run `python -m "
                   "src.realtime.infer --once` first, or pass ?mode=replay.",
        )
    # Flattened to the same top-level shape as the replay branch (asset,
    # method, target_position, price, reasoning, ...) plus `history` -- so
    # the frontend can use one response shape regardless of mode.
    return {"mode": "live", **decision, "history": history}


@app.get("/api/portfolio")
def get_portfolio(mode: ModeParam = Query("live")):
    if mode == "replay":
        assets = replay_mod.replay_portfolio(LIVE_ASSET_SYMBOLS)
        total_value = sum(pf["cash"] for pf in assets.values())
        return {"mode": "replay", "assets": assets, "total_portfolio_value": total_value}

    conn = livedb.get_connection()
    try:
        rows = livedb.get_all_portfolio_rows(conn)
    finally:
        conn.close()
    total_value = sum(r["cash"] for r in rows)
    return {
        "mode": "live",
        "assets": {r["asset"]: r for r in rows},
        "total_portfolio_value": total_value,
    }


@app.get("/api/backtest/{asset}")
def get_backtest(asset: str):
    """Phase 5 walk-forward results (docs/backtest_report.md), for
    side-by-side comparison against the live/replay decision above. Not
    mode-dependent -- these are fixed, already-computed historical
    numbers, not something a live vs replay toggle changes."""
    metrics = get_backtest_for_asset(asset)
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail=f"No walk-forward backtest results for {asset!r} in "
                   "docs/backtest_report.md (NFT collections were excluded from "
                   "Phase 5 -- no reliable price feed; see docs/feature_dictionary.md).",
        )
    return {"asset": asset, "methods": metrics}
