"""Real tests for the Phase 2 baselines (previously a placeholder --
`def test_placeholder(): assert True`). Uses small synthetic price/feature
CSVs written to a temp CWD (same pattern as
tests/data_pipeline/test_no_leakage.py) so these run fast and don't depend
on which assets happen to be fetched into data/.
"""
import numpy as np
import pandas as pd
import pytest

from src.baselines import momentum, supervised


N_DAYS = 120
SYMBOL = "TESTSYM"


def _write_price_csv(base_dir, close_prices):
    dates = pd.date_range("2023-01-01", periods=len(close_prices), freq="D")
    (base_dir / "data" / "raw" / "price").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"Close": close_prices}, index=dates)
    df.index.name = "date"
    df.to_csv(base_dir / "data/raw/price" / f"{SYMBOL}.csv")
    return dates


def _write_features_csv(base_dir, dates, n_features=4, seed=0):
    (base_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        rng.normal(size=(len(dates), n_features)),
        index=dates,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    df.index.name = "date"
    df.to_csv(base_dir / "data/processed" / f"{SYMBOL}_features.csv")


def test_calculate_metrics_reports_sharpe_sortino_drawdown():
    rng = np.random.RandomState(0)
    returns = pd.Series(rng.normal(0.001, 0.02, 500))
    metrics = momentum.calculate_metrics(returns)
    assert set(["Cumulative Return", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown"]).issubset(metrics.keys())
    assert np.isfinite(metrics["Sharpe Ratio"])
    assert np.isfinite(metrics["Sortino Ratio"])
    assert metrics["Max Drawdown"] <= 0


def test_calculate_metrics_handles_no_downside():
    # All-positive returns: downside deviation is undefined/zero -- must not
    # raise or return NaN/inf, should fall back to 0 for Sortino.
    returns = pd.Series([0.01, 0.02, 0.015, 0.03])
    metrics = momentum.calculate_metrics(returns)
    assert metrics["Sortino Ratio"] == 0


def test_momentum_strategy_runs_and_reports_date_range(tmp_path, monkeypatch):
    dates = _write_price_csv(tmp_path, 100 + np.cumsum(np.random.RandomState(1).normal(0, 1, N_DAYS)))
    monkeypatch.chdir(tmp_path)

    result = momentum.run_momentum_strategy(SYMBOL, window=20)

    assert result is not None
    assert "Sharpe Ratio" in result
    assert "Sortino Ratio" in result
    assert result["Start Date"] < result["End Date"]
    assert result["N Days"] > 0


def test_momentum_strategy_missing_price_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert momentum.run_momentum_strategy("NONEXISTENT") is None


def test_supervised_strategy_runs_with_time_series_split(tmp_path, monkeypatch):
    close_prices = 100 + np.cumsum(np.random.RandomState(2).normal(0, 1, N_DAYS))
    dates = _write_price_csv(tmp_path, close_prices)
    _write_features_csv(tmp_path, dates, seed=2)
    monkeypatch.chdir(tmp_path)

    result = supervised.run_supervised_strategy(SYMBOL)

    assert result is not None
    for key in ["Accuracy", "AUC", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown"]:
        assert key in result

    # Train must end strictly before test begins -- no overlap, no shuffling.
    assert result["Train End"] < result["Test Start"]


def test_supervised_strategy_missing_data_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert supervised.run_supervised_strategy("NONEXISTENT") is None
