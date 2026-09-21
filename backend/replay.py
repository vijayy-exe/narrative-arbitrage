"""backend/replay.py -- Phase 7 demo-resilience fallback.

Per the dev plan's Phase 7 fallback note: "Keep a --replay mode that
replays a recorded day from your walk-forward test set at accelerated
speed through the same web app, so the demo still looks live even if a
free API rate-limits you mid-presentation." This is a SAFEGUARD, not a
substitute for the real live loop (src/realtime/scheduler.py +
src/realtime/infer.py), which should still be run and shown working
separately. Every response this module produces is tagged "mode": "replay"
by the caller (backend/main.py) so it can never be mistaken for real live
data.

Replays each asset's REAL historical data -- data/processed/<symbol>_
features.csv joined against data/raw/price/<symbol>.csv, i.e. the exact
same processed data src/eval/walk_forward.py's rolling validation windows
were drawn from -- advancing one row further as real wall-clock time
passes, at an accelerated pace (default: one historical trading day
exposed every REPLAY_SECONDS_PER_DAY seconds of real time).

Decision logic is intentionally NOT reimplemented here: it reuses
src/realtime/infer.py's exact functions (apply_portfolio_step,
momentum_target_position, expanding_normalize, has_trained_policy) so a
replay decision is produced by the identical rule the live loop uses --
only the SOURCE of the "current" feature/price row differs (a historical
DataFrame row instead of live.db).
"""
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from src.agents.common import get_paths
from src.data_pipeline.features import recompute_derivatives_on_index
from src.env.trading_env import expanding_normalize
from src.realtime.infer import (
    INITIAL_CASH,
    apply_portfolio_step,
    default_portfolio_row,
    has_trained_policy,
    models_dir_for,
    momentum_target_position,
)

REPLAY_SECONDS_PER_DAY = 3.0


@lru_cache(maxsize=None)
def _historical_joined(symbol):
    features_path, price_path = get_paths(symbol)
    feat = pd.read_csv(features_path, index_col="date", parse_dates=True).sort_index()
    price = pd.read_csv(price_path, index_col="date", parse_dates=True).sort_index()
    full_df = feat.join(price[["Close"]], how="inner").sort_index()
    return recompute_derivatives_on_index(full_df)


class ReplayEngine:
    """One shared replay clock per backend process, started the first time
    it's used. Restarting the backend restarts the replay from day 0 of
    each asset's own history (different assets have different history
    lengths/start dates, so they will show different calendar dates at the
    same wall-clock moment -- an intentional signal that this is a replay,
    not a shared "today")."""

    def __init__(self, seconds_per_day=REPLAY_SECONDS_PER_DAY):
        self.start_wall = time.time()
        self.seconds_per_day = seconds_per_day
        self.portfolios = {}         # asset -> portfolio dict, evolves as replay advances
        self.decision_history = {}  # asset -> list of past decision dicts this replay run made

    def current_index(self, symbol):
        df = _historical_joined(symbol)
        elapsed = time.time() - self.start_wall
        idx = int(elapsed // self.seconds_per_day)
        return max(0, min(idx, len(df) - 1))

    def reset(self):
        self.start_wall = time.time()
        self.portfolios = {}
        self.decision_history = {}


_engine = ReplayEngine()


def get_engine():
    return _engine


def replay_features(symbol, history_len=60):
    df = _historical_joined(symbol)
    idx = _engine.current_index(symbol)
    feature_columns = [c for c in df.columns if c != "Close"]

    row = df.iloc[idx]
    values = {c: (None if pd.isna(row[c]) else float(row[c])) for c in feature_columns}

    start = max(0, idx - history_len + 1)
    hist_df = df.iloc[start: idx + 1]
    history = [
        {
            "date": d.date().isoformat(),
            "features": {c: (None if pd.isna(r[c]) else float(r[c])) for c in feature_columns},
        }
        for d, r in hist_df.iterrows()
    ]

    return {
        "asset": symbol,
        "date": df.index[idx].date().isoformat(),
        "features": values,
        "history": history,
        "replay_progress": {"index": idx, "total": len(df)},
    }


def replay_decision(symbol, models_dir="models"):
    df = _historical_joined(symbol)
    idx = _engine.current_index(symbol)
    window_df = df.iloc[: idx + 1]
    current_price = float(window_df["Close"].iloc[-1])

    prior = _engine.portfolios.get(symbol)

    if has_trained_policy(symbol, models_dir):
        from stable_baselines3 import PPO  # lazy import, heavy

        method = "ppo"
        row = prior or default_portfolio_row(method)
        feature_columns = [c for c in df.columns if c != "Close"]
        normalized = expanding_normalize(window_df[feature_columns])
        current_features = normalized.iloc[-1].to_numpy(dtype=np.float32)
        context = np.array([
            row["position"], row["cash"] / INITIAL_CASH, row["unrealized_pnl"] / INITIAL_CASH,
        ], dtype=np.float32)
        obs = np.concatenate([current_features, context])
        model = PPO.load(str(models_dir_for(symbol, models_dir)))
        raw_action, _ = model.predict(obs, deterministic=True)
        raw_action = float(np.asarray(raw_action).flatten()[0])
        target_position = float(np.clip(raw_action, -1.0, 1.0))
        reasoning = (
            f"[REPLAY] PPO policy (models/ppo_{symbol}/model.zip) as of "
            f"{window_df.index[-1].date()} -> raw action {raw_action:.4f} -> "
            f"clipped target position {target_position:.4f}."
        )
    else:
        method = "momentum_baseline"
        row = prior or default_portfolio_row(method)
        target_position = momentum_target_position(window_df["Close"])
        raw_action = target_position
        reasoning = (
            f"[REPLAY] No trained PPO checkpoint for {symbol} -- momentum baseline "
            f"(src/baselines/momentum.py's rule) as of {window_df.index[-1].date()} "
            f"-> target position {target_position:+.0f}. This is NOT an RL decision."
        )

    updated = apply_portfolio_step(row, target_position, current_price)
    updated["method"] = method
    _engine.portfolios[symbol] = updated

    result = {
        "asset": symbol,
        "method": method,
        "raw_action": raw_action,
        "target_position": target_position,
        "price": current_price,
        "reasoning": reasoning,
        "date": window_df.index[-1].date().isoformat(),
        "portfolio_value": updated["cash"],
        "replay_progress": {"index": idx, "total": len(df)},
    }
    hist = _engine.decision_history.setdefault(symbol, [])
    # Only append a new history entry when the replay clock has actually
    # advanced to a new day since the last poll -- otherwise a client
    # polling faster than REPLAY_SECONDS_PER_DAY would log duplicate
    # back-to-back entries for the same simulated day. Append a COPY (not
    # `result` itself) -- `result` gets a "history" key added right after
    # this, and appending the same object in would make each entry contain
    # itself (infinite recursion on JSON serialization).
    if not hist or hist[-1]["date"] != result["date"]:
        hist.append(dict(result))
    result["history"] = hist[-50:]
    return result


def replay_portfolio(symbols):
    """Ensures every asset has been "stepped" at least once (so a fresh
    replay run has a portfolio row before any decision has been polled for
    it yet) then returns the current simulated portfolio snapshot."""
    for symbol in symbols:
        if symbol not in _engine.portfolios:
            replay_decision(symbol)
    return {
        asset: {**pf} for asset, pf in _engine.portfolios.items() if asset in symbols
    }
