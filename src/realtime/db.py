"""Phase 7 -- SQLite schema and helpers for the live paper-trading loop.

A single local database (default `data/live/live.db`) holds three things:

- `live_features`: one row per (asset, date) with every attention-derivative
  column src/realtime/scheduler.py could compute that day, as a JSON blob
  (a blob rather than a wide fixed-column table because the set of columns
  is asset-dependent, e.g. crypto vs. stock; this mirrors how
  data/processed/<symbol>_features.csv already varies per asset).
- `live_prices`: one row per (asset, date) with the fetched Close price and
  which fetcher produced it.
- `decisions`: an append-only log of every decision src/realtime/infer.py
  makes, tagged with `method` ("ppo" or "momentum_baseline") -- NEVER
  blurred, per the project's honesty policy (see infer.py's module
  docstring).
- `portfolio`: current simulated cash/position/P&L per asset (one row per
  asset, overwritten each decision -- `decisions` is the append-only
  history of how it got there).
- `live_biz_signals`: one row per (asset, date) with the 4chan /biz/
  post_count/comment_count/sentiment signal from
  src/data_pipeline/fourchan_biz.py. DELIBERATELY KEPT OUT of
  `live_features`'s payload: `live_features` rows are read back by
  src/realtime/infer.py's build_ppo_observation() via `full_df.columns`
  (every key present in the payload, no fixed allowlist), so adding new
  keys there would silently change the PPO observation's dimensionality at
  inference time and break every trained checkpoint (all trained on the
  original data/processed/<symbol>_features.csv column set, which has no
  biz_* columns and never will for the historical window -- see
  fourchan_biz.py's module docstring on why it can't backfill). This table
  exists so the live /biz/ signal is captured and available for a FUTURE
  retraining run that deliberately extends the observation space, without
  silently corrupting current inference in the meantime.

No real brokerage/exchange orders are ever placed by anything that touches
this database.
"""
import json
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("data/live/live.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_features (
    asset TEXT NOT NULL,
    date TEXT NOT NULL,
    payload TEXT NOT NULL,
    sources_ok TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (asset, date)
);

CREATE TABLE IF NOT EXISTS live_prices (
    asset TEXT NOT NULL,
    date TEXT NOT NULL,
    close REAL,
    source TEXT,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (asset, date)
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    method TEXT NOT NULL,
    raw_action REAL,
    target_position REAL,
    price REAL,
    reasoning TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio (
    asset TEXT PRIMARY KEY,
    method TEXT NOT NULL,
    cash REAL NOT NULL,
    position REAL NOT NULL,
    entry_price REAL,
    last_price REAL,
    unrealized_pnl REAL NOT NULL,
    last_updated TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS live_biz_signals (
    asset TEXT NOT NULL,
    date TEXT NOT NULL,
    post_count REAL,
    comment_count REAL,
    sentiment REAL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (asset, date)
);
"""


def get_connection(db_path=None):
    # NOTE: resolves the DEFAULT_DB_PATH module global at CALL time (not as
    # a bound default-argument value at import time) so tests can
    # monkeypatch `src.realtime.db.DEFAULT_DB_PATH` and have every
    # no-argument get_connection() call (e.g. throughout backend/main.py)
    # honor the patched path.
    db_path = Path(db_path if db_path is not None else DEFAULT_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn):
    conn.executescript(SCHEMA)
    conn.commit()


# ─── live_features ─────────────────────────────────────────────────────────

def upsert_feature_row(conn, asset, date, payload: dict, sources_ok: dict, fetched_at: str):
    """`payload` values may be numpy scalars / NaN -- json.dumps needs plain
    Python floats, and NaN must become `null` (valid JSON), not the literal
    `NaN` token (technically emitted by default json.dumps, but not valid
    JSON and not parseable by every consumer, incl. some JS `JSON.parse`
    implementations the frontend may use)."""
    clean = {}
    for k, v in payload.items():
        if v is None:
            clean[k] = None
        else:
            fv = float(v)
            clean[k] = None if fv != fv else fv  # NaN != NaN
    conn.execute(
        "INSERT INTO live_features (asset, date, payload, sources_ok, fetched_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(asset, date) DO UPDATE SET payload=excluded.payload, "
        "sources_ok=excluded.sources_ok, fetched_at=excluded.fetched_at",
        (asset, str(date), json.dumps(clean), json.dumps(sources_ok), fetched_at),
    )
    conn.commit()


def get_latest_feature_row(conn, asset):
    row = conn.execute(
        "SELECT * FROM live_features WHERE asset=? ORDER BY date DESC LIMIT 1", (asset,)
    ).fetchone()
    if row is None:
        return None
    return {
        "asset": row["asset"],
        "date": row["date"],
        "features": json.loads(row["payload"]),
        "sources_ok": json.loads(row["sources_ok"]),
        "fetched_at": row["fetched_at"],
    }


def get_feature_history(conn, asset, limit=None):
    q = "SELECT * FROM live_features WHERE asset=? ORDER BY date ASC"
    params = (asset,)
    if limit is not None:
        q = "SELECT * FROM (" + q + ") ORDER BY date DESC LIMIT ?"
        params = (asset, limit)
    rows = conn.execute(q, params).fetchall()
    out = [
        {
            "asset": r["asset"],
            "date": r["date"],
            "features": json.loads(r["payload"]),
            "sources_ok": json.loads(r["sources_ok"]),
            "fetched_at": r["fetched_at"],
        }
        for r in rows
    ]
    if limit is not None:
        out = list(reversed(out))
    return out


# ─── live_prices ────────────────────────────────────────────────────────────

def upsert_price(conn, asset, date, close, source, fetched_at):
    conn.execute(
        "INSERT INTO live_prices (asset, date, close, source, fetched_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(asset, date) DO UPDATE SET close=excluded.close, "
        "source=excluded.source, fetched_at=excluded.fetched_at",
        (asset, str(date), None if close is None else float(close), source, fetched_at),
    )
    conn.commit()


def get_latest_price(conn, asset):
    row = conn.execute(
        "SELECT * FROM live_prices WHERE asset=? AND close IS NOT NULL ORDER BY date DESC LIMIT 1",
        (asset,),
    ).fetchone()
    return dict(row) if row is not None else None


def get_price_history(conn, asset):
    rows = conn.execute(
        "SELECT * FROM live_prices WHERE asset=? ORDER BY date ASC", (asset,)
    ).fetchall()
    return [dict(r) for r in rows]


# ─── decisions ──────────────────────────────────────────────────────────────

def insert_decision(conn, asset, timestamp, method, raw_action, target_position, price, reasoning):
    assert method in ("ppo", "momentum_baseline"), (
        f"method must be 'ppo' or 'momentum_baseline', got {method!r} -- "
        "never label a decision with a vaguer/blurrier method name, see "
        "src/realtime/infer.py module docstring."
    )
    conn.execute(
        "INSERT INTO decisions (asset, timestamp, method, raw_action, target_position, price, reasoning) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (asset, timestamp, method, raw_action, target_position, price, reasoning),
    )
    conn.commit()


def get_latest_decision(conn, asset):
    row = conn.execute(
        "SELECT * FROM decisions WHERE asset=? ORDER BY id DESC LIMIT 1", (asset,)
    ).fetchone()
    return dict(row) if row is not None else None


def get_decision_history(conn, asset, limit=100):
    rows = conn.execute(
        "SELECT * FROM decisions WHERE asset=? ORDER BY id DESC LIMIT ?", (asset, limit)
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


# ─── live_biz_signals (NOT fed into build_ppo_observation, see module
#     docstring) ────────────────────────────────────────────────────────────

def upsert_biz_signal(conn, asset, date, post_count, comment_count, sentiment, fetched_at):
    conn.execute(
        "INSERT INTO live_biz_signals (asset, date, post_count, comment_count, sentiment, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(asset, date) DO UPDATE SET post_count=excluded.post_count, "
        "comment_count=excluded.comment_count, sentiment=excluded.sentiment, "
        "fetched_at=excluded.fetched_at",
        (asset, str(date), float(post_count), float(comment_count), float(sentiment), fetched_at),
    )
    conn.commit()


def get_biz_signal_history(conn, asset, limit=None):
    q = "SELECT * FROM live_biz_signals WHERE asset=? ORDER BY date ASC"
    params = (asset,)
    if limit is not None:
        q = "SELECT * FROM (" + q + ") ORDER BY date DESC LIMIT ?"
        params = (asset, limit)
    rows = conn.execute(q, params).fetchall()
    out = [dict(r) for r in rows]
    if limit is not None:
        out = list(reversed(out))
    return out


# ─── portfolio ──────────────────────────────────────────────────────────────

def get_portfolio_row(conn, asset):
    row = conn.execute("SELECT * FROM portfolio WHERE asset=?", (asset,)).fetchone()
    return dict(row) if row is not None else None


def get_all_portfolio_rows(conn):
    rows = conn.execute("SELECT * FROM portfolio ORDER BY asset ASC").fetchall()
    return [dict(r) for r in rows]


def upsert_portfolio_row(conn, asset, method, cash, position, entry_price, last_price,
                          unrealized_pnl, last_updated):
    conn.execute(
        "INSERT INTO portfolio (asset, method, cash, position, entry_price, last_price, "
        "unrealized_pnl, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(asset) DO UPDATE SET method=excluded.method, cash=excluded.cash, "
        "position=excluded.position, entry_price=excluded.entry_price, "
        "last_price=excluded.last_price, unrealized_pnl=excluded.unrealized_pnl, "
        "last_updated=excluded.last_updated",
        (asset, method, cash, position, entry_price, last_price, unrealized_pnl, last_updated),
    )
    conn.commit()
