"""FastAPI endpoint tests, following the repo's existing convention of never
hitting live external APIs. All of these exercise the REPLAY data path
(backend/replay.py, which reads real historical data/processed + data/raw/
price files already checked into this repo -- no network) rather than the
live path (which needs data/live/live.db to be populated by
src/realtime/scheduler.py/infer.py first, and is covered separately by
tests/realtime/). Live-mode 404 behavior on an empty DB is also checked
directly, in isolation, so it doesn't depend on the real data/live/live.db's
contents.
"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app, LIVE_ASSET_SYMBOLS

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_assets_shapes_and_honest_method_labels():
    resp = client.get("/api/assets")
    assert resp.status_code == 200
    data = resp.json()
    assets = {a["symbol"]: a for a in data["assets"]}
    assert set(assets) == set(LIVE_ASSET_SYMBOLS)
    # All 5 basket assets have a real trained checkpoint as of 2026-09-15 --
    # see src/realtime/infer.py's has_trained_policy(), which checks disk
    # dynamically rather than hardcoding DOGE.
    for sym in LIVE_ASSET_SYMBOLS:
        assert assets[sym]["method"] == "ppo"


@pytest.mark.parametrize("asset", ["GME", "AMC", "DOGE", "SHIB", "PEPE"])
def test_features_replay_mode_well_shaped(asset):
    resp = client.get(f"/api/features/{asset}?mode=replay")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "replay"
    assert data["asset"] == asset
    assert "date" in data
    assert isinstance(data["features"], dict)
    assert len(data["features"]) > 0


@pytest.mark.parametrize("asset", ["GME", "AMC", "DOGE", "SHIB", "PEPE"])
def test_decision_replay_mode_well_shaped_and_honest(asset):
    resp = client.get(f"/api/decision/{asset}?mode=replay")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "replay"
    # All 5 basket assets have a real trained checkpoint as of 2026-09-15 --
    # backend/replay.py checks has_trained_policy() dynamically, so every
    # asset in the basket now honestly gets a "ppo" decision.
    assert data["method"] == "ppo"
    assert -1.0 <= data["target_position"] <= 1.0
    assert isinstance(data["price"], float) and data["price"] > 0


def test_portfolio_replay_mode_well_shaped():
    resp = client.get("/api/portfolio?mode=replay")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "replay"
    assert set(data["assets"]) == set(LIVE_ASSET_SYMBOLS)
    for asset, pf in data["assets"].items():
        assert "cash" in pf and "position" in pf
    assert isinstance(data["total_portfolio_value"], float)


def test_backtest_endpoint_returns_phase5_walk_forward_numbers():
    resp = client.get("/api/backtest/DOGE")
    assert resp.status_code == 200
    data = resp.json()
    assert data["asset"] == "DOGE"
    assert set(data["methods"]) == {"momentum", "ppo", "supervised"}
    assert "Sharpe Ratio" in data["methods"]["ppo"]


def test_backtest_endpoint_404_for_asset_never_backtested():
    resp = client.get("/api/backtest/boredapeyachtclub")
    assert resp.status_code == 404


def test_unknown_asset_returns_404():
    resp = client.get("/api/features/NOT_A_REAL_ASSET")
    assert resp.status_code == 404
    resp = client.get("/api/decision/NOT_A_REAL_ASSET?mode=replay")
    assert resp.status_code == 404


def test_live_mode_404s_cleanly_on_empty_db(tmp_path, monkeypatch):
    """Isolates the live path from whatever the real data/live/live.db
    happens to contain by pointing src.realtime.db's default path at a
    fresh, empty tmp_path db."""
    from src.realtime import db as livedb

    monkeypatch.setattr(livedb, "DEFAULT_DB_PATH", tmp_path / "empty_live.db")

    resp = client.get("/api/features/DOGE?mode=live")
    assert resp.status_code == 404
    resp = client.get("/api/decision/DOGE?mode=live")
    assert resp.status_code == 404

    resp = client.get("/api/portfolio?mode=live")
    assert resp.status_code == 200
    assert resp.json()["assets"] == {}
