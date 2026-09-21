"""Tests for src/eval/pareto_frontier.py (Tier 2, Phase 3)."""
import numpy as np
import pandas as pd
import pytest

from src.eval.pareto_frontier import REFERENCE_POINTS, sweep_asset


def _write_tiny_asset(tmp_path, monkeypatch, symbol="PARETOTEST", n_days=280, seed=0):
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


def test_reference_points_cover_the_full_basket():
    assert set(REFERENCE_POINTS.keys()) == {"DOGE", "GME", "AMC", "PEPE", "SHIB"}
    for asset, points in REFERENCE_POINTS.items():
        assert "Original PPO" in points
        assert "CVaR (alpha=0.2, original)" in points


def test_sweep_asset_returns_one_row_per_alpha(tmp_path, monkeypatch):
    symbol = _write_tiny_asset(tmp_path, monkeypatch)
    alphas = [0.9, 0.3]
    results = sweep_asset(symbol, alphas, ppo_timesteps=64, seed=0, limit_rolls=1)
    assert len(results) == len(alphas)
    assert set(results["cvar_alpha"]) == set(alphas)
    for col in ["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Cumulative Return"]:
        assert results[col].apply(np.isfinite).all()
