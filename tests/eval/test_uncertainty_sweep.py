"""Tests for src/eval/uncertainty_sweep.py (Tier 2 k-sweep follow-up)."""
import numpy as np
import pandas as pd

from src.eval.uncertainty_sweep import REFERENCE_POINTS, sweep_asset


def _write_tiny_asset(tmp_path, monkeypatch, symbol="UNCSWEEPTEST", n_days=280, seed=0):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "raw" / "price").mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rng = np.random.RandomState(seed)

    pd.DataFrame({"Close": 100 + np.cumsum(rng.normal(0, 1, n_days))}, index=dates) \
        .rename_axis("date").to_csv(tmp_path / "data/raw/price" / f"{symbol}.csv")
    pd.DataFrame(rng.normal(size=(n_days, 3)), index=dates, columns=["a", "b", "c"]) \
        .rename_axis("date").to_csv(tmp_path / "data/processed" / f"{symbol}_features.csv")
    return symbol


def test_reference_points_cover_gme_and_amc():
    assert set(REFERENCE_POINTS.keys()) == {"GME", "AMC"}
    for asset, points in REFERENCE_POINTS.items():
        assert "Original PPO (no scaling)" in points
        assert "Uncertainty-aware (k=2.0, original single-k run)" in points


def test_sweep_asset_returns_one_row_per_k_reusing_one_trained_pair(tmp_path, monkeypatch):
    symbol = _write_tiny_asset(tmp_path, monkeypatch)
    ks = [0.5, 2.0, 4.0]
    results = sweep_asset(symbol, ks, n_critic_members=2, ppo_timesteps=64, seed=0, limit_rolls=1)
    assert len(results) == len(ks)
    assert set(results["k"]) == set(ks)
    for col in ["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Cumulative Return"]:
        assert results[col].apply(np.isfinite).all()
