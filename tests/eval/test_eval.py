"""Tests for Phase 5's walk-forward harness (previously a placeholder).
Pure-function tests for window generation/leakage-audit/trade-stats run
fast with no I/O; the end-to-end test at the bottom uses a tiny synthetic
asset with a deliberately tiny ppo_timesteps budget so it stays fast.
"""
import numpy as np
import pandas as pd
import pytest

from src.eval.walk_forward import (
    RL_ROLL_EVALUATORS,
    audit_windows_no_leakage,
    compute_trade_stats,
    generate_rolling_windows,
    run_walk_forward_for_asset,
)


# ─── generate_rolling_windows ────────────────────────────────────────────────


def test_generate_rolling_windows_basic_shape():
    index = pd.date_range("2023-01-01", periods=400, freq="D")
    windows = generate_rolling_windows(index, train_months=6, val_months=1, step_months=1)
    assert len(windows) > 0
    for train_start, train_end, val_start, val_end in windows:
        assert train_start < train_end < val_start <= val_end
        assert train_start >= index.min()
        assert val_end <= index.max()


def test_generate_rolling_windows_rolls_forward():
    index = pd.date_range("2023-01-01", periods=400, freq="D")
    windows = generate_rolling_windows(index, train_months=6, val_months=1, step_months=1)
    val_starts = [w[2] for w in windows]
    assert val_starts == sorted(val_starts)
    assert len(set(val_starts)) == len(val_starts), "consecutive rolls must validate on different windows"


def test_generate_rolling_windows_too_short_history_gives_zero_windows():
    index = pd.date_range("2023-01-01", periods=30, freq="D")  # 1 month, need 7
    windows = generate_rolling_windows(index, train_months=6, val_months=1, step_months=1)
    assert windows == []


# ─── audit_windows_no_leakage ────────────────────────────────────────────────


def test_audit_passes_on_valid_windows():
    index = pd.date_range("2023-01-01", periods=400, freq="D")
    windows = generate_rolling_windows(index, train_months=6, val_months=1, step_months=1)
    assert audit_windows_no_leakage(windows) is True


def test_audit_catches_train_overlapping_validation():
    bad_windows = [(
        pd.Timestamp("2023-01-01"), pd.Timestamp("2023-07-05"),  # train ends AFTER val starts
        pd.Timestamp("2023-07-01"), pd.Timestamp("2023-07-31"),
    )]
    with pytest.raises(AssertionError):
        audit_windows_no_leakage(bad_windows)


def test_audit_catches_validation_moving_backwards():
    bad_windows = [
        (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-30"), pd.Timestamp("2023-07-01"), pd.Timestamp("2023-07-31")),
        (pd.Timestamp("2023-02-01"), pd.Timestamp("2023-07-30"), pd.Timestamp("2023-08-01"), pd.Timestamp("2023-07-15")),
    ]
    with pytest.raises(AssertionError):
        audit_windows_no_leakage(bad_windows)


# ─── compute_trade_stats ─────────────────────────────────────────────────────


def test_compute_trade_stats_counts_runs_not_days():
    # 3 long days, 2 short days, 1 flat day -> 2 trades.
    positions = [1, 1, 1, -1, -1, 0]
    returns =   [0.01, 0.01, -0.005, 0.02, -0.01, 0.0]
    stats = compute_trade_stats(positions, returns)
    assert stats["n_trades"] == 2
    assert stats["avg_trade_duration"] == pytest.approx((3 + 2) / 2)


def test_compute_trade_stats_win_rate():
    # Trade 1 (long, 2 days): net positive. Trade 2 (short, 2 days): net negative.
    positions = [1, 1, -1, -1]
    returns = [0.05, 0.05, 0.03, 0.03]  # long trade: +10.25%; short trade * -1 sign handled by caller convention
    stats = compute_trade_stats(positions, returns)
    assert stats["n_trades"] == 2
    assert 0.0 <= stats["win_rate"] <= 1.0


def test_compute_trade_stats_no_trades_returns_nan():
    stats = compute_trade_stats([0, 0, 0], [0.01, -0.01, 0.02])
    assert stats["n_trades"] == 0
    assert np.isnan(stats["win_rate"])
    assert np.isnan(stats["avg_trade_duration"])


# ─── end-to-end smoke test ───────────────────────────────────────────────────


def test_walk_forward_runs_end_to_end_on_tiny_asset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "price").mkdir(parents=True)

    n_days = 280  # ~9 months: enough for one 6mo-train/1mo-val roll, maybe two
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rng = np.random.RandomState(0)

    close_prices = 100 + np.cumsum(rng.normal(0, 1, n_days))
    df_price = pd.DataFrame({"Close": close_prices}, index=dates)
    df_price.index.name = "date"
    df_price.to_csv(tmp_path / "data/raw/price/TESTSYM.csv")

    df_features = pd.DataFrame(rng.normal(size=(n_days, 3)), index=dates, columns=["a", "b", "c"])
    df_features.index.name = "date"
    df_features.to_csv(tmp_path / "data/processed/TESTSYM_features.csv")

    results = run_walk_forward_for_asset(
        "TESTSYM", train_months=6, val_months=1, step_months=1,
        ppo_timesteps=64, seed=0, limit_rolls=1,
    )

    assert not results.empty
    assert set(results["method"]) <= {"momentum", "supervised", "ppo"}
    for col in ["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Cumulative Return"]:
        assert results[col].apply(np.isfinite).all()


# ─── rl_variant (Upgrades 1-3 walk-forward extension, added 2026-09-20) ────


def _write_tiny_asset(tmp_path, monkeypatch, symbol="TESTSYM2", n_days=280, seed=1):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "raw" / "price").mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rng = np.random.RandomState(seed)

    close_prices = 100 + np.cumsum(rng.normal(0, 1, n_days))
    df_price = pd.DataFrame({"Close": close_prices}, index=dates)
    df_price.index.name = "date"
    df_price.to_csv(tmp_path / "data/raw/price" / f"{symbol}.csv")

    df_features = pd.DataFrame(rng.normal(size=(n_days, 3)), index=dates, columns=["a", "b", "c"])
    df_features.index.name = "date"
    df_features.to_csv(tmp_path / "data/processed" / f"{symbol}_features.csv")
    return symbol


def test_rl_roll_evaluators_covers_all_seven_upgrade_variants():
    assert set(RL_ROLL_EVALUATORS.keys()) == {
        "ppo", "cvar", "lstm", "transformer", "lstm_ensemble", "bc_pretrained", "uncertainty_aware",
    }


@pytest.mark.parametrize("variant", [
    "ppo", "cvar", "lstm", "transformer", "lstm_ensemble", "bc_pretrained", "uncertainty_aware",
])
def test_run_walk_forward_for_asset_works_for_every_rl_variant(tmp_path, monkeypatch, variant):
    symbol = _write_tiny_asset(tmp_path, monkeypatch)
    results = run_walk_forward_for_asset(
        symbol, train_months=6, val_months=1, step_months=1,
        ppo_timesteps=64, seed=0, limit_rolls=1, rl_variant=variant,
    )
    assert not results.empty
    assert "ppo" in set(results["method"])  # always labeled "ppo" regardless of rl_variant, see docstring
    for col in ["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Cumulative Return"]:
        assert results[col].apply(np.isfinite).all()


def test_run_walk_forward_for_asset_no_baselines_skips_momentum_and_supervised(tmp_path, monkeypatch):
    symbol = _write_tiny_asset(tmp_path, monkeypatch)
    results = run_walk_forward_for_asset(
        symbol, train_months=6, val_months=1, step_months=1,
        ppo_timesteps=64, seed=0, limit_rolls=1, rl_variant="ppo", include_baselines=False,
    )
    assert set(results["method"]) == {"ppo"}
