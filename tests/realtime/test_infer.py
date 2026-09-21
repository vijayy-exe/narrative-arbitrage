"""Tests for src/realtime/infer.py.

Two integration paths are covered, mirroring the honesty split the module
itself enforces:

1. The momentum-baseline path, exercised end-to-end against a small
   SYNTHETIC asset (isolated in tmp_path, chdir'd so the hardcoded
   data/processed / data/raw/price relative paths resolve there instead of
   the real repo data) that deliberately has NO models/ppo_<symbol>/
   checkpoint -- this must automatically fall back to momentum_baseline,
   never fabricate a "ppo" decision.
2. The real PPO path, exercised against the REAL DOGE checkpoint
   (models/ppo_DOGE/model.zip) and REAL historical DOGE data already in
   this repo (data/processed/DOGE_features.csv, data/raw/price/DOGE.csv) --
   this is the one asset honestly allowed to produce a "ppo"-tagged
   decision. Uses a throwaway SQLite db under tmp_path so it never touches
   the real data/live/live.db.
"""
import numpy as np
import pandas as pd
import pytest

from src.realtime import db as livedb
from src.realtime import infer


# ─── pure-function tests: portfolio accounting / momentum rule ────────────

def test_apply_portfolio_step_charges_fee_on_entry():
    row = infer.default_portfolio_row("momentum_baseline")
    updated = infer.apply_portfolio_step(row, target_position=1.0, current_price=100.0,
                                          initial_cash=10_000.0, transaction_fee=0.01)
    # Opening a full position with a 1% fee costs 1% of notional immediately,
    # even before any price move is realized (no last_price yet -> step_return=0).
    assert updated["cash"] == pytest.approx(10_000.0 - 100.0)
    assert updated["position"] == 1.0
    assert updated["entry_price"] == 100.0


def test_apply_portfolio_step_realizes_pnl_on_price_move():
    row = infer.default_portfolio_row("ppo")
    row = infer.apply_portfolio_step(row, target_position=1.0, current_price=100.0, transaction_fee=0.0)
    row = infer.apply_portfolio_step(row, target_position=1.0, current_price=110.0, transaction_fee=0.0)
    # +10% move while fully long (fraction-of-initial_cash sizing) -> +10% of initial_cash.
    assert row["cash"] == pytest.approx(10_000.0 * 1.10)
    assert row["unrealized_pnl"] == pytest.approx(10_000.0 * 0.10)


def test_momentum_target_position_matches_momentum_py_rule():
    # Strictly increasing series -> positive trailing return -> long.
    up = pd.Series(np.linspace(100, 200, 30))
    assert infer.momentum_target_position(up, window=20) == 1.0

    down = pd.Series(np.linspace(200, 100, 30))
    assert infer.momentum_target_position(down, window=20) == -1.0


def test_momentum_target_position_defaults_short_on_insufficient_history():
    # Fewer than `window` rows -> pct_change is NaN -> NaN>0 is False ->
    # falls back to -1, exactly matching src/baselines/momentum.py's
    # (pre-existing, unchanged) np.where(NaN > 0, 1, -1) behavior.
    short = pd.Series([100.0, 101.0, 102.0])
    assert infer.momentum_target_position(short, window=20) == -1.0


def test_has_trained_policy_true_for_all_checkpointed_assets():
    # As of 2026-09-15 all 5 basket assets have a models/ppo_<symbol>/model.zip
    # checkpoint (DOGE was first, 2026-09-07; AMC/GME/PEPE/SHIB followed) --
    # has_trained_policy() checks disk dynamically, so it must be True for
    # all of them, not just DOGE.
    for symbol in ("DOGE", "GME", "AMC", "SHIB", "PEPE"):
        assert infer.has_trained_policy(symbol) is True
    assert infer.has_trained_policy("NOT_A_REAL_ASSET") is False


# ─── end-to-end: momentum-baseline asset (synthetic, isolated) ─────────────

def _write_synthetic_asset(base_dir, symbol, n_days=60):
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    prices = 100 + np.cumsum(np.random.RandomState(0).normal(0, 1, n_days))

    processed_dir = base_dir / "data" / "processed"
    price_dir = base_dir / "data" / "raw" / "price"
    processed_dir.mkdir(parents=True, exist_ok=True)
    price_dir.mkdir(parents=True, exist_ok=True)

    feat = pd.DataFrame({"pageviews_velocity": np.zeros(n_days)}, index=dates)
    feat.index.name = "date"
    feat.to_csv(processed_dir / f"{symbol}_features.csv")

    price = pd.DataFrame({"Close": prices}, index=dates)
    price.index.name = "date"
    price.to_csv(price_dir / f"{symbol}.csv")
    return dates, prices


def test_decide_and_update_uses_momentum_baseline_when_no_checkpoint_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_synthetic_asset(tmp_path, "SYN")
    # No models/ppo_SYN/model.zip anywhere under tmp_path -> must not be "ppo".

    conn = livedb.get_connection(tmp_path / "data" / "live" / "live.db")
    result = infer.decide_and_update("SYN", conn)

    assert result["method"] == "momentum_baseline"
    assert result["target_position"] in (-1.0, 1.0)

    decision = livedb.get_latest_decision(conn, "SYN")
    assert decision["method"] == "momentum_baseline"
    assert "momentum" in decision["reasoning"].lower()
    assert "not an rl decision" in decision["reasoning"].lower()

    portfolio = livedb.get_portfolio_row(conn, "SYN")
    assert portfolio["method"] == "momentum_baseline"
    conn.close()


def test_decide_and_update_never_labels_synthetic_asset_as_ppo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_synthetic_asset(tmp_path, "SYN2")
    conn = livedb.get_connection(tmp_path / "data" / "live" / "live.db")
    for _ in range(3):
        result = infer.decide_and_update("SYN2", conn)
        assert result["method"] != "ppo"
    conn.close()


# ─── end-to-end: the real, trained DOGE PPO checkpoint ─────────────────────

def test_decide_and_update_uses_real_ppo_checkpoint_for_doge(tmp_path):
    """Runs against the REAL repo data/model (no chdir), only the SQLite db
    is redirected to tmp_path so this never touches data/live/live.db."""
    conn = livedb.get_connection(tmp_path / "live.db")
    result = infer.decide_and_update("DOGE", conn)

    assert result["method"] == "ppo"
    assert np.isfinite(result["target_position"])
    assert -1.0 <= result["target_position"] <= 1.0
    assert "models/ppo_DOGE/model.zip" in result["reasoning"]

    decision = livedb.get_latest_decision(conn, "DOGE")
    assert decision["method"] == "ppo"

    portfolio = livedb.get_portfolio_row(conn, "DOGE")
    assert portfolio["method"] == "ppo"
    assert np.isfinite(portfolio["cash"])
    conn.close()


def test_build_ppo_observation_handles_none_valued_live_features(tmp_path):
    """Regression test for a real bug found 2026-09-20, the first time the
    live scheduler was ever run against real APIs: Wikipedia pageviews are
    always unavailable for "today" (the Wikimedia API lags 1-2 days), so a
    live feature row with a None-valued column is the NORMAL case on every
    live run, not a rare edge case -- yet no existing test ever inserted a
    live row with a None value before calling build_ppo_observation (the
    other tests above all use an empty live.db, which takes the
    historical-only code path and never exercises _live_rows_as_frame's
    None-handling at all). Before the fix, this raised
    'TypeError: unsupported operand type(s) for -: NoneType and int'
    inside recompute_derivatives_on_index's Series.diff() call."""
    conn = livedb.get_connection(tmp_path / "live.db")
    feat = pd.read_csv("data/processed/DOGE_features.csv", index_col="date", parse_dates=True)
    raw_cols = [c for c in feat.columns if "_velocity" not in c and "_acceleration" not in c
                and c not in ("edit_war_ratio", "sentiment_divergence")]
    live_row = {c: 123.0 for c in raw_cols}
    live_row["pageviews"] = None  # simulates a failed/unavailable fetch, e.g. today's Wikipedia pageviews

    livedb.upsert_feature_row(
        conn, "DOGE", "2099-01-01", live_row,
        {"wikipedia": False, "trends": True, "hackernews": True, "price": True},
        "2099-01-01T00:00:00+00:00",
    )
    livedb.upsert_price(conn, "DOGE", "2099-01-01", 0.5, "ccxt_binance", "2099-01-01T00:00:00+00:00")

    row = infer.default_portfolio_row("ppo")
    obs, price, as_of = infer.build_ppo_observation("DOGE", conn, row)
    assert np.isfinite(obs).all()  # None -> NaN -> expanding_normalize's fillna(0.0), never a crash
    conn.close()


def test_build_ppo_observation_shape_matches_env_feature_count(tmp_path):
    conn = livedb.get_connection(tmp_path / "live.db")
    row = infer.default_portfolio_row("ppo")
    obs, price, as_of = infer.build_ppo_observation("DOGE", conn, row)

    feat = pd.read_csv("data/processed/DOGE_features.csv")
    n_feature_cols = len(feat.columns) - 1  # minus the 'date' column
    assert obs.shape == (n_feature_cols + 3,)  # + [position, cash, unrealized_pnl]
    assert np.isfinite(obs).all()
    assert price > 0
    conn.close()
