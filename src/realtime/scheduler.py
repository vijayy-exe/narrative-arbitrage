"""Phase 7 -- live data-refresh scheduler.

python -m src.realtime.scheduler --once                  # single fetch cycle (for cron/testing)
python -m src.realtime.scheduler --interval-hours 6       # run forever, refresh every N hours

On each run, for every asset in LIVE_ASSETS (the meme-stock + small-cap-crypto
basket from configs/assets.yaml -- NFT collections are excluded here for the
same reason they're excluded from RL training/backtesting: no reliable price
feed was ever wired for them, see src/agents/common.py's available_assets()):

  1. Fetches "today"'s attention data using the EXISTING Phase 1 fetchers --
     src/data_pipeline/wikipedia.py (fetch_pageviews, fetch_edits),
     src/data_pipeline/trends.py (fetch_trends), src/data_pipeline/
     hackernews.py (fetch_hn_stories). Reddit (src/data_pipeline/reddit.py)
     is deliberately NOT called -- it is a documented, currently-blocked gap
     (no self-serve API credentials obtainable as of 2026-09, see
     docs/feature_dictionary.md) and this module must not pretend otherwise.
     src/data_pipeline/fourchan_biz.py (fetch_biz_snapshot) is ALSO called,
     as a genuinely working (if historically un-backfillable) live signal --
     but its output is written to its own live_biz_signals table, never
     merged into RAW_COLUMNS/the feature payload; see fourchan_biz.py's and
     src/realtime/db.py's docstrings for why (it would silently change the
     trained PPO models' expected observation dimensionality otherwise).
  2. Fetches today's price -- yfinance for stocks (meme_stocks), ccxt's
     Binance public REST endpoint for crypto (small_cap_crypto) -- via
     small wrappers around src/data_pipeline/price.py's own fetch functions.
  3. Recomputes derived (velocity/acceleration/edit_war_ratio) features
     using src/data_pipeline/features.py's OWN functions
     (recompute_derivatives_on_index, compute_edit_war_ratio) -- never
     reimplemented here.
  4. Appends everything to data/live/live.db (src/realtime/db.py).

RESILIENCE: every external call is wrapped by `_safe_call()`. If one source
fails (network error, rate limit, whatever) that source's fields are simply
left absent (NaN) for today and `sources_ok` records the failure -- the run
continues updating every other source rather than crashing. This is checked
by tests/realtime/test_scheduler.py (one source mocked to fail, others must
still be written).
"""
import argparse
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import yaml

from src.data_pipeline import wikipedia, trends, hackernews, fourchan_biz
from src.data_pipeline.features import recompute_derivatives_on_index, compute_edit_war_ratio
from src.realtime import db as livedb

logger = logging.getLogger("realtime.scheduler")

# Raw signal columns this scheduler can populate (the subset of
# docs/feature_dictionary.md's "raw/passthrough" columns that DON'T require
# Reddit -- pageviews/edit_count/unique_editors/revert_count from Wikipedia,
# search_interest from Trends, hn_* from Hacker News). `post_count`,
# `comment_count`, `sentiment` (Reddit-only) are never populated here.
RAW_COLUMNS = [
    "pageviews", "edit_count", "unique_editors", "revert_count",
    "search_interest", "hn_story_count", "hn_points", "hn_comment_count",
]


def load_assets_config():
    with open("configs/assets.yaml", "r") as f:
        return yaml.safe_load(f)


def live_assets(assets_config=None):
    """The live-loop basket: meme_stocks + small_cap_crypto only (real,
    working price feeds -- see module docstring). Each entry carries a
    normalized 'symbol', 'category' ('stock'|'crypto'), 'wikipedia_title',
    'name' and resolved 'hn_query'.
    """
    assets_config = assets_config or load_assets_config()
    out = []
    for stock in assets_config.get("meme_stocks", []):
        out.append({
            "symbol": stock["ticker"],
            "category": "stock",
            "name": stock.get("name"),
            "wikipedia_title": stock.get("wikipedia_title"),
            "hn_query": stock.get("hn_query") or stock.get("name"),
            "biz_query": stock.get("biz_query") or stock.get("name"),
        })
    for crypto in assets_config.get("small_cap_crypto", []):
        out.append({
            "symbol": crypto["symbol"],
            "category": "crypto",
            "name": crypto.get("name"),
            "wikipedia_title": crypto.get("wikipedia_title"),
            "hn_query": crypto.get("hn_query") or crypto.get("name"),
            "biz_query": crypto.get("biz_query") or crypto.get("name"),
        })
    return out


def _safe_call(source_name, symbol, fn, *args, **kwargs):
    """Runs fn(*args, **kwargs); on ANY exception, logs it and returns
    (None, False) instead of propagating -- the one mechanism that makes the
    whole scheduler resilient to a single source failing (dev plan Phase 7:
    "keep the whole loop resilient to a single source failing... log the gap
    rather than crashing the scheduler")."""
    try:
        return fn(*args, **kwargs), True
    except Exception as e:  # noqa: BLE001 -- deliberately broad: ANY failure
        # from a live external API must not crash the scheduler.
        logger.warning("live fetch failed: source=%s asset=%s error=%s: %s",
                        source_name, symbol, type(e).__name__, e)
        return None, False


# ─── per-source fetch wrappers (today only) ────────────────────────────────

def fetch_wikipedia_today(title, asof):
    """Returns a dict of {pageviews, edit_count, unique_editors,
    revert_count} for `asof` (a date string), or an empty dict on failure/no
    data. Reuses wikipedia.py's own fetch_pageviews/fetch_edits verbatim.
    """
    result = {}
    views, ok_v = _safe_call("wikipedia_pageviews", title, wikipedia.fetch_pageviews, title, asof, asof)
    if ok_v and views is not None and not views.empty and asof in [d.isoformat() for d in views.index]:
        result["pageviews"] = float(views.loc[pd.to_datetime(asof).date(), "pageviews"])

    edits, ok_e = _safe_call("wikipedia_edits", title, wikipedia.fetch_edits, title, asof, asof)
    if ok_e and edits is not None and not edits.empty:
        idx = pd.to_datetime(asof).date()
        if idx in edits.index:
            row = edits.loc[idx]
            result["edit_count"] = float(row.get("edit_count", 0.0))
            result["unique_editors"] = float(row.get("unique_editors", 0.0))
            result["revert_count"] = float(row.get("revert_count", 0.0))
    return result, (ok_v and ok_e)


def fetch_trends_today(name, asof):
    """Google Trends is weekly, not daily (see docs/feature_dictionary.md).
    Fetches a short trailing window and forward-fills to `asof`, exactly
    the causal convention src/data_pipeline/features.py's process_asset()
    uses for the historical backfill -- never zero-filled.
    """
    start = (pd.to_datetime(asof) - pd.Timedelta(days=13)).date().isoformat()
    df, ok = _safe_call("trends", name, trends.fetch_trends, name, start, asof)
    if not ok or df is None or df.empty:
        return {}, ok
    df = df.sort_index()
    asof_ts = pd.to_datetime(asof)
    df = df[df.index <= asof_ts]
    if df.empty:
        return {}, ok
    return {"search_interest": float(df["search_interest"].iloc[-1])}, ok


def fetch_biz_today(query, asof):
    """4chan /biz/ live snapshot (src/data_pipeline/fourchan_biz.py) --
    NOT part of RAW_COLUMNS / the PPO observation (see src/realtime/db.py's
    live_biz_signals docstring for why). `asof` is accepted for interface
    symmetry with the other fetch_*_today wrappers but unused: /biz/ has no
    date-range query, every call reflects the board's current live state,
    not a specific historical day."""
    result, ok = _safe_call("fourchan_biz", query, fourchan_biz.fetch_biz_snapshot, query)
    if not ok:
        return {}, ok
    return result, ok


def fetch_hn_today(query, asof):
    rows, ok = _safe_call("hackernews", query, hackernews.fetch_hn_stories, query, asof, asof)
    if not ok:
        return {}, ok
    if not rows:
        # A genuine zero-hit day is a real value, not a fetch failure --
        # same distinction docs/feature_dictionary.md draws for the
        # historical hn_* columns.
        return {"hn_story_count": 0.0, "hn_points": 0.0, "hn_comment_count": 0.0}, ok
    return {
        "hn_story_count": float(len(rows)),
        "hn_points": float(sum(r["points"] for r in rows)),
        "hn_comment_count": float(sum(r["num_comments"] for r in rows)),
    }, ok


def fetch_stock_price_today(ticker):
    """yfinance, ~15-20 min delayed -- irrelevant here since features update
    daily (dev plan Phase 7 free-data-sources note)."""
    import yfinance as yf

    def _fetch():
        data = yf.download(ticker, period="5d", progress=False)
        if data.empty:
            raise ValueError(f"yfinance returned no rows for {ticker}")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return float(data["Close"].iloc[-1]), data.index[-1].date().isoformat()

    result, ok = _safe_call("yfinance_price", ticker, _fetch)
    if not ok:
        return None, None, ok
    close, price_date = result
    return close, price_date, ok


def fetch_crypto_price_today(symbol):
    """ccxt's Binance PUBLIC REST endpoint -- free, no API key needed for
    market data (dev plan Phase 7)."""
    import ccxt

    def _fetch():
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol + "/USDT", timeframe="1d", limit=2)
        if not ohlcv:
            raise ValueError(f"ccxt returned no candles for {symbol}/USDT")
        last = ohlcv[-1]
        close = float(last[4])
        price_date = pd.to_datetime(last[0], unit="ms").date().isoformat()
        return close, price_date

    result, ok = _safe_call("ccxt_binance_price", symbol, _fetch)
    if not ok:
        return None, None, ok
    close, price_date = result
    return close, price_date, ok


# ─── feature recomputation (reuses features.py, never reimplements it) ────

def _historical_last_level_row(symbol):
    """Falls back to data/processed/<symbol>_features.csv's last row (RAW_COLUMNS
    only) as "yesterday" when live.db has no prior row yet for this asset --
    i.e. the very first live run after a gap since the Phase 1 backfill."""
    path = Path(f"data/processed/{symbol}_features.csv")
    if not path.exists():
        return None, None
    df = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
    cols = [c for c in RAW_COLUMNS if c in df.columns]
    if not cols or df.empty:
        return None, None
    return df.index[-1], df[cols].iloc[-1].to_dict()


def compute_today_row(conn, symbol, raw_today: dict, asof: str):
    """Builds a 2-row frame [previous known raw levels, today's raw levels]
    and recomputes velocity/acceleration/edit_war_ratio on it via
    src/data_pipeline/features.py's own recompute_derivatives_on_index() /
    compute_edit_war_ratio() -- exactly the function every other consumer
    (NarrativeTradingEnv, walk_forward.py) uses to recompute derivatives
    after joining processed features against a price index (see that
    function's docstring). A raw field missing from EITHER day (fetch
    failure, or a genuinely first-ever run with no prior day) yields NaN
    for that day's velocity/acceleration, not a fabricated 0 -- this is
    treated the same as "no history yet" by NarrativeTradingEnv's/infer.py's
    expanding-normalize step (NaN -> neutral 0.0 in the OBSERVATION, not the
    stored feature row).
    """
    prev_date, prev_row = None, None
    existing = livedb.get_feature_history(conn, symbol, limit=1)
    if existing:
        prev_date = pd.to_datetime(existing[-1]["date"])
        prev_row = {k: existing[-1]["features"].get(k) for k in RAW_COLUMNS}
    else:
        prev_date, prev_row = _historical_last_level_row(symbol)

    today_date = pd.to_datetime(asof)
    rows = {}
    if prev_row is not None and prev_date is not None and prev_date < today_date:
        rows[prev_date] = prev_row
    rows[today_date] = {c: raw_today.get(c) for c in RAW_COLUMNS}

    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    frame.index.name = "date"
    frame = recompute_derivatives_on_index(frame)
    frame = compute_edit_war_ratio(frame)

    today_row = frame.loc[today_date].to_dict()
    return today_row


# ─── orchestration ──────────────────────────────────────────────────────────

def refresh_asset(conn, asset, asof=None):
    """Runs one full refresh cycle for a single asset dict (see
    live_assets()). Returns a dict summarizing what succeeded, for logging/
    tests -- never raises on a source failure (see _safe_call).
    """
    symbol = asset["symbol"]
    asof = asof or datetime.now(timezone.utc).date().isoformat()
    fetched_at = datetime.now(timezone.utc).isoformat()

    sources_ok = {"wikipedia": False, "trends": False, "hackernews": False, "price": False, "biz": False}
    raw_today = {}

    if asset.get("wikipedia_title"):
        wiki_fields, ok = fetch_wikipedia_today(asset["wikipedia_title"], asof)
        raw_today.update(wiki_fields)
        sources_ok["wikipedia"] = ok

    if asset.get("name"):
        trend_fields, ok = fetch_trends_today(asset["name"], asof)
        raw_today.update(trend_fields)
        sources_ok["trends"] = ok

    if asset.get("hn_query"):
        hn_fields, ok = fetch_hn_today(asset["hn_query"], asof)
        raw_today.update(hn_fields)
        sources_ok["hackernews"] = ok

    # Deliberately NOT merged into raw_today / RAW_COLUMNS / upsert_feature_row
    # -- written to its own live_biz_signals table instead. See
    # src/realtime/db.py's live_biz_signals docstring: mixing this into the
    # payload build_ppo_observation() reads would silently change the
    # trained-model observation dimensionality.
    if asset.get("biz_query"):
        biz_fields, ok = fetch_biz_today(asset["biz_query"], asof)
        sources_ok["biz"] = ok
        if ok and biz_fields:
            livedb.upsert_biz_signal(
                conn, symbol, asof,
                biz_fields.get("post_count", 0.0),
                biz_fields.get("comment_count", 0.0),
                biz_fields.get("sentiment", 0.0),
                fetched_at,
            )

    # Wrapped in _safe_call HERE too (not just inside each fetch_*_price_
    # today function): this is the actual resilience boundary refresh_asset
    # promises -- a price-fetch function raising directly (a fetcher's own
    # internal safety wrapper failing, or a future price source that forgets
    # to wrap itself) must still not crash the whole asset's refresh.
    if asset["category"] == "stock":
        price_result, price_call_ok = _safe_call("stock_price", symbol, fetch_stock_price_today, symbol)
    else:
        price_result, price_call_ok = _safe_call("crypto_price", symbol, fetch_crypto_price_today, symbol)
    if price_call_ok and price_result is not None:
        close, price_date, ok = price_result
    else:
        close, price_date, ok = None, None, False
    sources_ok["price"] = ok
    if ok and close is not None:
        livedb.upsert_price(conn, symbol, price_date or asof, close,
                             "yfinance" if asset["category"] == "stock" else "ccxt_binance",
                             fetched_at)

    today_row = compute_today_row(conn, symbol, raw_today, asof)
    livedb.upsert_feature_row(conn, symbol, asof, today_row, sources_ok, fetched_at)

    failed = [k for k, v in sources_ok.items() if not v]
    if failed:
        logger.info("%s: refreshed with degraded sources (failed: %s)", symbol, failed)
    else:
        logger.info("%s: refreshed, all sources OK", symbol)

    return {"symbol": symbol, "asof": asof, "sources_ok": sources_ok, "price_ok": ok}


def run_once(db_path=livedb.DEFAULT_DB_PATH, asof=None, assets=None):
    conn = livedb.get_connection(db_path)
    try:
        assets = assets if assets is not None else live_assets()
        results = []
        for asset in assets:
            results.append(refresh_asset(conn, asset, asof=asof))
            time.sleep(0.2)  # be polite across assets, matching Phase 1 fetchers
        return results
    finally:
        conn.close()


def start_scheduler(interval_hours=6, db_path=livedb.DEFAULT_DB_PATH):
    """Runs run_once() immediately, then every `interval_hours` hours,
    forever -- the dev plan's "a background job (APScheduler...) that
    refreshes live data... every N hours" (section 10). Uses a
    BlockingScheduler since this is meant to run as its own standalone
    process (invoked via `python -m src.realtime.scheduler`), not embedded
    inside the FastAPI process -- keeping the scheduler and the API server
    as separate, independently-restartable processes is simpler to operate
    and debug for a local demo than sharing one process/event loop.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler = BlockingScheduler()
    scheduler.add_job(lambda: run_once(db_path=db_path), IntervalTrigger(hours=interval_hours),
                       next_run_time=datetime.now(timezone.utc))
    logger.info("Starting live scheduler: refresh every %s hour(s), db=%s", interval_hours, db_path)
    scheduler.start()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run a single fetch cycle and exit")
    parser.add_argument("--interval-hours", type=float, default=6.0)
    parser.add_argument("--db-path", default=str(livedb.DEFAULT_DB_PATH))
    parser.add_argument("--asof", default=None, help="Override 'today' (YYYY-MM-DD), for testing")
    args = parser.parse_args()

    if args.once:
        results = run_once(db_path=args.db_path, asof=args.asof)
        for r in results:
            print(r)
    else:
        start_scheduler(interval_hours=args.interval_hours, db_path=args.db_path)


if __name__ == "__main__":
    main()
