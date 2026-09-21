"""Tests for src/agents/hparam_search.py (Upgrade 4). A pure-function unit
test for the search-space suggestion helper, plus a tiny (not-realistic-
budget) end-to-end sweep + multiseed smoke test against a synthetic asset,
following the same tiny-budget-smoke-test convention as the other Upgrade
test files in this directory.
"""
import numpy as np
import optuna
import pandas as pd
import pytest

from src.agents import hparam_search
from src.agents.common import get_paths


N_DAYS = 60


def _write_synthetic_asset(monkeypatch, tmp_path, symbol="HPTEST", n_days=N_DAYS, seed=0):
    monkeypatch.chdir(tmp_path)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rng = np.random.RandomState(seed)

    processed_dir = tmp_path / "data" / "processed"
    price_dir = tmp_path / "data" / "raw" / "price"
    processed_dir.mkdir(parents=True, exist_ok=True)
    price_dir.mkdir(parents=True, exist_ok=True)

    feat = pd.DataFrame(rng.normal(size=(n_days, 3)), index=dates, columns=["a", "b", "c"])
    feat.index.name = "date"
    feat.to_csv(processed_dir / f"{symbol}_features.csv")

    price = pd.DataFrame({"Close": 100 + np.cumsum(rng.normal(size=n_days))}, index=dates)
    price.index.name = "date"
    price.to_csv(price_dir / f"{symbol}.csv")
    return symbol


def test_suggest_params_covers_every_search_space_key():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = hparam_search._suggest_params(trial)
    assert set(params.keys()) == set(hparam_search.SEARCH_SPACE.keys())
    assert 1e-5 <= params["learning_rate"] <= 1e-3
    assert 1e-4 <= params["ent_coef"] <= 5e-2
    assert 0.80 <= params["gae_lambda"] <= 0.99
    assert params["lstm_hidden_size"] in [32, 64, 128]


def test_train_and_eval_lstm_runs_and_returns_finite_metrics(tmp_path, monkeypatch):
    symbol = _write_synthetic_asset(monkeypatch, tmp_path)
    params = {"learning_rate": 3e-4, "ent_coef": 0.01, "gae_lambda": 0.95, "lstm_hidden_size": 32}
    model, metrics, momentum = hparam_search.train_and_eval_lstm(symbol, params, total_timesteps=64, seed=0)
    assert np.isfinite(metrics["Sharpe Ratio"]) or metrics["Sharpe Ratio"] == 0.0
    assert model is not None


def test_run_sweep_completes_and_returns_a_study(tmp_path, monkeypatch):
    symbol = _write_synthetic_asset(monkeypatch, tmp_path)
    study = hparam_search.run_sweep(symbol, n_trials=2, timesteps_per_trial=64, seed=0)
    assert len(study.trials) == 2
    assert set(study.best_params.keys()) == set(hparam_search.SEARCH_SPACE.keys())


def test_run_multiseed_returns_one_row_per_seed_and_summary_has_mean_std(tmp_path, monkeypatch):
    symbol = _write_synthetic_asset(monkeypatch, tmp_path)
    params = {"learning_rate": 3e-4, "ent_coef": 0.01, "gae_lambda": 0.95, "lstm_hidden_size": 32}
    rows, momentum = hparam_search.run_multiseed(symbol, params, n_seeds=2, total_timesteps=64,
                                                  models_dir=str(tmp_path / "models_final"))
    assert len(rows) == 2
    assert rows[0]["seed"] != rows[1]["seed"]

    summary = hparam_search.summarize_multiseed(rows, momentum)
    assert set(summary.keys()) == {"Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Cumulative Return"}
    for stats in summary.values():
        assert "mean" in stats and "std" in stats
        assert np.isfinite(stats["mean"])
        assert stats["std"] >= 0.0
