"""Phase 7 -- live decision loop: turns the live feature/price store into a
paper-trading decision per asset, and updates a simulated portfolio. NO REAL
ORDERS ARE EVER PLACED -- this only writes rows to data/live/live.db.

python -m src.realtime.infer --once [--asset DOGE ...]

HONESTY BOUNDARY (this is the whole point of keeping this module small and
explicit, not a stylistic choice):

  - DOGE is the ONLY asset with an actual trained PPO checkpoint
    (models/ppo_DOGE/model.zip, produced by Phase 4a's src/agents/train_ppo.py
    -- see repo state notes / docs/backtest_report.md). It is the only asset
    that gets a real RL-policy-driven decision here, tagged
    method="ppo" in the `decisions`/`portfolio` tables.
  - Every other live-loop asset (GME, AMC, SHIB, PEPE) has NO trained
    checkpoint. Rather than fabricate one, or silently skip them, this
    module runs the ALREADY-IMPLEMENTED momentum baseline
    (src/baselines/momentum.py's exact rule: position = +1 if the trailing
    20-day return is positive, else -1) on their live+historical price
    series, tagged method="momentum_baseline". This is never blurred with
    "ppo" anywhere -- not in the DB schema (src/realtime/db.py asserts
    method is one of exactly these two strings), not in reasoning text, and
    the FastAPI layer (backend/) passes this field straight through.
  - Which asset gets which method is determined by whether
    models/ppo_<symbol>/model.zip actually EXISTS on disk (has_trained_policy()),
    not a hardcoded asset name -- so if a checkpoint is trained for another
    asset later, it is picked up automatically and honestly, without a code
    change silently going stale.

OBSERVATION CONSTRUCTION: mirrors NarrativeTradingEnv's own observation
exactly -- same feature columns, same expanding (causal) z-score
normalization (src/env/trading_env.py's expanding_normalize(), imported not
reimplemented), same portfolio-context layout ([position, cash/initial_cash,
unrealized_pnl/initial_cash]) -- built from the historical processed/price
CSVs (Phase 1 backfill) concatenated with whatever live rows
src/realtime/scheduler.py has appended to data/live/live.db so far. There
will typically be a real date GAP between the historical backfill's end date
and "today" (the live loop only fetches days it actually runs on, it does
not backfill the gap) -- derivative features spanning that gap are
mathematically well-defined but describe a multi-year change, not a
day-over-day one; this is a known, documented characteristic of a live demo
resuming long after a historical backfill, not a bug.
"""
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.common import get_paths
from src.data_pipeline.features import recompute_derivatives_on_index
from src.env.trading_env import expanding_normalize
from src.realtime import db as livedb
from src.realtime.scheduler import live_assets

logger = logging.getLogger("realtime.infer")

INITIAL_CASH = 10_000.0          # matches NarrativeTradingEnv's default
TRANSACTION_FEE = 0.001          # matches NarrativeTradingEnv's default
MOMENTUM_WINDOW = 20             # matches src/baselines/momentum.py's default


def models_dir_for(symbol, models_dir="models"):
    return Path(models_dir) / f"ppo_{symbol}" / "model.zip"


def has_trained_policy(symbol, models_dir="models"):
    return models_dir_for(symbol, models_dir).exists()


# ─── shared portfolio accounting (mirrors NarrativeTradingEnv.step()) ──────

def default_portfolio_row(method):
    return {
        "method": method, "cash": INITIAL_CASH, "position": 0.0,
        "entry_price": None, "last_price": None, "unrealized_pnl": 0.0,
    }


def apply_portfolio_step(row, target_position, current_price,
                          initial_cash=INITIAL_CASH, transaction_fee=TRANSACTION_FEE):
    """Same notional (CFD-style), price-scale-invariant accounting as
    NarrativeTradingEnv.step() -- position is a FRACTION of initial_cash,
    `cash` IS total portfolio value, realized P&L is credited for the
    price move that happened while the OLD position was held (not the new
    one), and a transaction cost is charged on the dollar notional
    rebalanced. See src/env/trading_env.py's step() for the full derivation/
    bug-history comments this mirrors -- deliberately not re-derived here.
    """
    old_position = row["position"]
    old_cash = row["cash"]
    last_price = row.get("last_price")

    cost = abs(target_position - old_position) * initial_cash * transaction_fee

    if last_price is not None and last_price > 0:
        pct_change = (current_price - last_price) / last_price
        step_return = old_position * pct_change * initial_cash
    else:
        step_return = 0.0

    new_cash = old_cash + step_return - cost

    if target_position != old_position:
        new_entry_price = current_price if target_position != 0 else None
    else:
        new_entry_price = row.get("entry_price")

    if target_position != 0 and new_entry_price is not None:
        unrealized_pnl = target_position * initial_cash * (current_price - new_entry_price) / new_entry_price
    else:
        unrealized_pnl = 0.0

    return {
        "method": row["method"], "cash": new_cash, "position": target_position,
        "entry_price": new_entry_price, "last_price": current_price,
        "unrealized_pnl": unrealized_pnl,
    }


# ─── momentum baseline decision (reuses momentum.py's exact rule) ──────────

def momentum_target_position(price_series, window=MOMENTUM_WINDOW):
    """Replicates src/baselines/momentum.py's run_momentum_strategy() rule
    EXACTLY: position = +1 if the trailing `window`-day pct change is
    positive, else -1 -- including its (pre-existing, unchanged here)
    behavior on insufficient history: pandas' pct_change(window) is NaN
    until `window` rows exist, and `NaN > 0` is False in both numpy and
    plain Python, so np.where(...) / this reimplementation both fall back to
    -1 (short) rather than "no position" for those early rows. Not fixed
    here since fixing it would make this a DIFFERENT rule than the one
    src/baselines/momentum.py was actually backtested with (docs/
    backtest_report.md's momentum numbers reflect this exact behavior).
    """
    momentum = price_series.pct_change(window)
    last = momentum.iloc[-1] if len(momentum) else float("nan")
    return 1.0 if (pd.notna(last) and last > 0) else -1.0


def get_price_series_for_momentum(symbol, conn):
    """Historical price CSV (Phase 1 backfill) concatenated with whatever
    live prices scheduler.py has appended so far -- price only, no
    attention features, exactly matching momentum.py's inputs."""
    _, price_path = get_paths(symbol)
    hist = pd.read_csv(price_path, index_col="date", parse_dates=True).sort_index()["Close"]
    live_rows = livedb.get_price_history(conn, symbol)
    if live_rows:
        live_series = pd.Series(
            {pd.to_datetime(r["date"]): r["close"] for r in live_rows if r["close"] is not None}
        ).sort_index()
        combined = pd.concat([hist, live_series])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined
    return hist


# ─── PPO observation construction (reuses trading_env's normalization) ─────

def _load_historical_joined(symbol):
    features_path, price_path = get_paths(symbol)
    feat = pd.read_csv(features_path, index_col="date", parse_dates=True).sort_index()
    price = pd.read_csv(price_path, index_col="date", parse_dates=True).sort_index()
    return feat.join(price[["Close"]], how="inner").sort_index()


def _live_rows_as_frame(conn, symbol):
    """Converts whatever live.db has for `symbol` (features joined with
    price on date) into a DataFrame shaped like _load_historical_joined()'s
    output, restricted to dates that actually have BOTH a live feature row
    and a live price (an inner join, same convention NarrativeTradingEnv
    uses for the historical CSVs).

    BUG FIX (found 2026-09-20, the first time this function was ever
    exercised against a REAL live scheduler run rather than synthetic test
    fixtures): a raw feature that failed to fetch (e.g. Wikipedia pageviews
    for "today" -- the Wikimedia pageviews API always lags 1-2 days behind
    real time, so this is the NORMAL case on every live run, not a rare
    edge case) is stored in live.db's JSON payload as `null`, which
    json.loads() decodes to Python `None`, not `float('nan')`. A DataFrame
    built directly from those `None` values has an `object`-dtype column
    for that feature, and `Series.diff()` (called by
    recompute_derivatives_on_index() a few lines below) raises
    `TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'`
    the moment it hits one -- crashing every live decision for any asset
    with an unfetched source that day. Every consumer downstream
    (recompute_derivatives_on_index, expanding_normalize) already expects
    missing values as NaN, not None (see e.g. upsert_feature_row's own
    None/NaN handling on the way IN) -- this was simply never converted
    back on the way OUT. Fixed by coercing None -> np.nan explicitly here.
    """
    feature_rows = livedb.get_feature_history(conn, symbol)
    price_rows = {r["date"]: r["close"] for r in livedb.get_price_history(conn, symbol) if r["close"] is not None}
    records = {}
    for fr in feature_rows:
        if fr["date"] in price_rows:
            rec = {c: (v if v is not None else np.nan) for c, v in fr["features"].items()}
            rec["Close"] = price_rows[fr["date"]]
            records[pd.to_datetime(fr["date"])] = rec
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(records, orient="index").sort_index()


def build_ppo_observation(symbol, conn, portfolio_row):
    """Builds the exact observation NarrativeTradingEnv would produce for
    `symbol`'s current (last) row: historical + live feature/price rows,
    re-differenced (recompute_derivatives_on_index -- not reimplemented),
    causally z-scored (expanding_normalize -- not reimplemented), plus the
    live portfolio context, in that order (matching
    NarrativeTradingEnv._get_observation()).
    """
    historical = _load_historical_joined(symbol)
    live = _live_rows_as_frame(conn, symbol)
    full_df = pd.concat([historical, live]) if not live.empty else historical
    full_df = full_df[~full_df.index.duplicated(keep="last")].sort_index()

    full_df = recompute_derivatives_on_index(full_df)
    feature_columns = [c for c in full_df.columns if c != "Close"]
    normalized = expanding_normalize(full_df[feature_columns])
    current_features = normalized.iloc[-1].to_numpy(dtype=np.float32)

    context = np.array([
        portfolio_row["position"],
        portfolio_row["cash"] / INITIAL_CASH,
        portfolio_row["unrealized_pnl"] / INITIAL_CASH,
    ], dtype=np.float32)

    obs = np.concatenate([current_features, context])
    current_price = float(full_df["Close"].iloc[-1])
    as_of = full_df.index[-1]
    return obs, current_price, as_of


# ─── orchestration ──────────────────────────────────────────────────────────

def decide_and_update(symbol, conn, models_dir="models"):
    """Runs one decision cycle for `symbol` and persists it (decision log +
    portfolio row). Returns a summary dict. Never places a real order."""
    now = datetime.now(timezone.utc).isoformat()
    prior = livedb.get_portfolio_row(conn, symbol)

    if has_trained_policy(symbol, models_dir):
        from stable_baselines3 import PPO  # imported lazily: heavy, and only needed here

        method = "ppo"
        row = prior or default_portfolio_row(method)
        row["method"] = method
        obs, current_price, as_of = build_ppo_observation(symbol, conn, row)
        model = PPO.load(str(models_dir_for(symbol, models_dir)))
        raw_action, _ = model.predict(obs, deterministic=True)
        raw_action = float(np.asarray(raw_action).flatten()[0])
        target_position = float(np.clip(raw_action, -1.0, 1.0))
        reasoning = (
            f"PPO policy (models/ppo_{symbol}/model.zip) on normalized live+historical "
            f"attention features as of {as_of.date()} -> raw action {raw_action:.4f} "
            f"-> clipped target position {target_position:.4f}."
        )
    else:
        method = "momentum_baseline"
        row = prior or default_portfolio_row(method)
        row["method"] = method
        price_series = get_price_series_for_momentum(symbol, conn)
        current_price = float(price_series.iloc[-1])
        as_of = price_series.index[-1]
        target_position = momentum_target_position(price_series)
        raw_action = target_position
        reasoning = (
            f"No trained PPO checkpoint for {symbol} -- using the momentum baseline "
            f"(src/baselines/momentum.py's rule: {MOMENTUM_WINDOW}-day trailing return "
            f"sign) on live+historical price as of {as_of.date() if hasattr(as_of, 'date') else as_of} "
            f"-> target position {target_position:+.0f}. This is NOT an RL decision."
        )

    updated = apply_portfolio_step(row, target_position, current_price)
    livedb.upsert_portfolio_row(
        conn, symbol, updated["method"], updated["cash"], updated["position"],
        updated["entry_price"], updated["last_price"], updated["unrealized_pnl"], now,
    )
    livedb.insert_decision(conn, symbol, now, method, raw_action, target_position, current_price, reasoning)

    logger.info("%s [%s]: target_position=%.4f price=%.6f portfolio_value=%.2f",
                symbol, method, target_position, current_price, updated["cash"])

    return {
        "asset": symbol, "method": method, "target_position": target_position,
        "price": current_price, "portfolio_value": updated["cash"], "reasoning": reasoning,
    }


def run_once(db_path=livedb.DEFAULT_DB_PATH, assets=None, models_dir="models"):
    conn = livedb.get_connection(db_path)
    try:
        symbols = assets if assets is not None else [a["symbol"] for a in live_assets()]
        return [decide_and_update(sym, conn, models_dir=models_dir) for sym in symbols]
    finally:
        conn.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", nargs="+", default=None, help="Default: all live-loop assets")
    parser.add_argument("--db-path", default=str(livedb.DEFAULT_DB_PATH))
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    results = run_once(db_path=args.db_path, assets=args.asset, models_dir=args.models_dir)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
