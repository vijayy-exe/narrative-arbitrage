"""Tests for src/realtime/scheduler.py. No live network calls -- every
external fetcher is monkeypatched at the module-function level, following
the same convention as tests/data_pipeline/test_data.py (mock the fetch
functions, assert on the shape/contents of what gets written), so these
run in CI without live API access.
"""
import pandas as pd
import pytest

from src.realtime import db as livedb
from src.realtime import scheduler


def test_live_assets_basket_excludes_nfts():
    assets = scheduler.live_assets()
    by_symbol = {a["symbol"]: a["category"] for a in assets}
    # Meme stocks + small-cap crypto only -- NFTs have no live price feed
    # wired (same reason src/agents/common.py's available_assets() excludes
    # them from RL training/backtesting).
    assert by_symbol == {
        "GME": "stock", "AMC": "stock",
        "PEPE": "crypto", "DOGE": "crypto", "SHIB": "crypto",
    }


def test_safe_call_catches_any_exception_and_reports_failure():
    def boom():
        raise RuntimeError("simulated network/rate-limit failure")

    result, ok = scheduler._safe_call("fake_source", "SYM", boom)
    assert result is None
    assert ok is False


def test_safe_call_passes_through_a_successful_result():
    result, ok = scheduler._safe_call("fake_source", "SYM", lambda: 42)
    assert result == 42
    assert ok is True


def test_refresh_asset_is_resilient_to_a_single_source_failure(tmp_path, monkeypatch):
    """Core Phase 7 resilience requirement: if Wikipedia fails, Trends/HN/
    price must still be fetched and written, not skipped."""
    conn = livedb.get_connection(tmp_path / "live.db")

    def _boom_pageviews(*a, **k):
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(scheduler.wikipedia, "fetch_pageviews", _boom_pageviews)
    monkeypatch.setattr(scheduler.wikipedia, "fetch_edits", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        scheduler.trends, "fetch_trends",
        lambda name, start, end: pd.DataFrame({"search_interest": [42.0]}, index=pd.to_datetime([end])),
    )
    monkeypatch.setattr(
        scheduler.hackernews, "fetch_hn_stories",
        lambda query, start, end, **k: [{"points": 5, "num_comments": 2}],
    )
    monkeypatch.setattr(scheduler, "fetch_stock_price_today", lambda ticker: (123.45, "2026-09-12", True))

    asset = {
        "symbol": "GME", "category": "stock", "name": "GameStop",
        "wikipedia_title": "GameStop", "hn_query": "GameStop stock",
    }

    result = scheduler.refresh_asset(conn, asset, asof="2026-09-12")

    # asset dict has no "biz_query" -> the /biz/ fetch is skipped entirely
    # (not attempted, not a failure) -- sources_ok["biz"] stays at its
    # initialized False.
    assert result["sources_ok"] == {
        "wikipedia": False, "trends": True, "hackernews": True, "price": True, "biz": False,
    }

    row = livedb.get_latest_feature_row(conn, "GME")
    assert row is not None
    assert row["date"] == "2026-09-12"
    assert row["features"]["pageviews"] is None          # wikipedia failed -> not fabricated
    assert row["features"]["search_interest"] == 42.0
    assert row["features"]["hn_story_count"] == 1.0
    assert row["features"]["hn_points"] == 5.0

    price_row = livedb.get_latest_price(conn, "GME")
    assert price_row is not None
    assert price_row["close"] == pytest.approx(123.45)
    assert price_row["source"] == "yfinance"

    conn.close()


def test_refresh_asset_marks_price_failure_without_crashing(tmp_path, monkeypatch):
    conn = livedb.get_connection(tmp_path / "live.db")

    monkeypatch.setattr(scheduler.wikipedia, "fetch_pageviews", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(scheduler.wikipedia, "fetch_edits", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(scheduler.trends, "fetch_trends", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(scheduler.hackernews, "fetch_hn_stories", lambda *a, **k: [])

    def _boom_price(symbol):
        raise RuntimeError("ccxt network error")

    monkeypatch.setattr(scheduler, "fetch_crypto_price_today", _boom_price)

    asset = {
        "symbol": "DOGE", "category": "crypto", "name": "Dogecoin",
        "wikipedia_title": "Dogecoin", "hn_query": "Dogecoin",
    }
    result = scheduler.refresh_asset(conn, asset, asof="2026-09-12")

    assert result["sources_ok"]["price"] is False
    assert livedb.get_latest_price(conn, "DOGE") is None  # nothing fabricated
    # but the feature row (hn/trends/wiki -- all "empty but not failed" here) still gets written
    row = livedb.get_latest_feature_row(conn, "DOGE")
    assert row is not None
    conn.close()


def test_refresh_asset_writes_biz_signal_to_its_own_table_not_the_feature_payload(tmp_path, monkeypatch):
    """The 4chan /biz/ signal (src/data_pipeline/fourchan_biz.py) must land in
    live_biz_signals, and must NEVER appear as a key in live_features'
    payload -- see src/realtime/db.py's live_biz_signals docstring: mixing
    it into the feature payload would silently change build_ppo_observation()'s
    output dimensionality and break every trained PPO checkpoint."""
    conn = livedb.get_connection(tmp_path / "live.db")

    monkeypatch.setattr(scheduler.wikipedia, "fetch_pageviews", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(scheduler.wikipedia, "fetch_edits", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(scheduler.trends, "fetch_trends", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(scheduler.hackernews, "fetch_hn_stories", lambda *a, **k: [])
    monkeypatch.setattr(scheduler, "fetch_stock_price_today", lambda ticker: (123.45, "2026-09-20", True))
    monkeypatch.setattr(
        scheduler.fourchan_biz, "fetch_biz_snapshot",
        lambda query, **k: {"post_count": 4.0, "comment_count": 37.0, "sentiment": -0.12},
    )

    asset = {
        "symbol": "GME", "category": "stock", "name": "GameStop",
        "wikipedia_title": "GameStop", "hn_query": "GameStop stock", "biz_query": "GME",
    }
    result = scheduler.refresh_asset(conn, asset, asof="2026-09-20")

    assert result["sources_ok"]["biz"] is True

    biz_history = livedb.get_biz_signal_history(conn, "GME")
    assert len(biz_history) == 1
    assert biz_history[0]["post_count"] == pytest.approx(4.0)
    assert biz_history[0]["comment_count"] == pytest.approx(37.0)
    assert biz_history[0]["sentiment"] == pytest.approx(-0.12)

    feature_row = livedb.get_latest_feature_row(conn, "GME")
    assert "post_count" not in feature_row["features"]
    assert "comment_count" not in feature_row["features"]
    assert "sentiment" not in feature_row["features"]
    conn.close()


def test_refresh_asset_skips_biz_fetch_when_no_biz_query_configured(tmp_path, monkeypatch):
    conn = livedb.get_connection(tmp_path / "live.db")

    monkeypatch.setattr(scheduler.wikipedia, "fetch_pageviews", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(scheduler.wikipedia, "fetch_edits", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(scheduler.trends, "fetch_trends", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(scheduler.hackernews, "fetch_hn_stories", lambda *a, **k: [])
    monkeypatch.setattr(scheduler, "fetch_stock_price_today", lambda ticker: (123.45, "2026-09-20", True))

    def _boom(*a, **k):
        raise AssertionError("fetch_biz_snapshot should not be called when biz_query is absent")

    monkeypatch.setattr(scheduler.fourchan_biz, "fetch_biz_snapshot", _boom)

    asset = {"symbol": "GME", "category": "stock", "name": "GameStop", "wikipedia_title": "GameStop"}
    result = scheduler.refresh_asset(conn, asset, asof="2026-09-20")

    assert result["sources_ok"]["biz"] is False
    assert livedb.get_biz_signal_history(conn, "GME") == []
    conn.close()


def test_compute_today_row_computes_velocity_from_previous_live_row(tmp_path):
    """Pure recomputation test, isolated from any real data/processed/ file
    (symbol 'ZZZ' has no such file) -- exercises the
    recompute_derivatives_on_index / compute_edit_war_ratio reuse path."""
    conn = livedb.get_connection(tmp_path / "live.db")
    livedb.upsert_feature_row(
        conn, "ZZZ", "2026-09-10",
        {
            "pageviews": 100.0, "edit_count": 2.0, "unique_editors": 1.0, "revert_count": 0.0,
            "search_interest": 10.0, "hn_story_count": 0.0, "hn_points": 0.0, "hn_comment_count": 0.0,
        },
        {"wikipedia": True, "trends": True, "hackernews": True, "price": True},
        "2026-09-10T00:00:00+00:00",
    )

    today_row = scheduler.compute_today_row(
        conn, "ZZZ",
        {
            "pageviews": 150.0, "edit_count": 3.0, "unique_editors": 2.0, "revert_count": 1.0,
            "search_interest": 10.0, "hn_story_count": 1.0, "hn_points": 5.0, "hn_comment_count": 2.0,
        },
        "2026-09-11",
    )

    assert today_row["pageviews"] == pytest.approx(150.0)
    assert today_row["pageviews_velocity"] == pytest.approx(50.0)
    assert today_row["edit_count_velocity"] == pytest.approx(1.0)
    assert today_row["edit_war_ratio"] == pytest.approx(1 / 4)  # revert=1 / (edit=3 + 1)
    conn.close()


def test_compute_today_row_leaves_missing_source_as_nan_not_zero(tmp_path):
    conn = livedb.get_connection(tmp_path / "live.db")
    today_row = scheduler.compute_today_row(
        conn, "ZZZ_NEW",
        {"pageviews": None, "edit_count": None, "unique_editors": None, "revert_count": None,
         "search_interest": 5.0, "hn_story_count": 0.0, "hn_points": 0.0, "hn_comment_count": 0.0},
        "2026-09-11",
    )
    assert pd.isna(today_row["pageviews"])
    assert today_row["search_interest"] == pytest.approx(5.0)
    conn.close()
