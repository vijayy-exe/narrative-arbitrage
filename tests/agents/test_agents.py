"""Tests for Phase 4a (previously a placeholder). Two things are covered:

1. classify_action_distribution() (src/eval/reward_hacking_check.py) as a
   pure-function unit test -- no training required, fast.
2. A real, but deliberately tiny, PPO training + rollout run against
   NarrativeTradingEnv, to catch integration breakage between the env's
   observation/action spaces and stable-baselines3 (e.g. dtype mismatches,
   NaN propagation) that a pure-function test can't. Uses n_steps/
   total_timesteps small enough to run in a couple of seconds, not a real
   training budget -- this is a wiring smoke test, not a "does it learn"
   test (Phase 5's walk-forward backtest is where performance is judged).
"""
import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.train_ppo import (
    ValidationSharpeCallback,
    make_env,
    momentum_metrics_on_window,
    portfolio_values_to_returns,
    rollout_deterministic,
)
from src.eval.reward_hacking_check import classify_action_distribution


N_DAYS = 60


def _write_env_csvs(tmp_path, close_prices, seed=0):
    dates = pd.date_range("2023-01-01", periods=len(close_prices), freq="D")
    rng = np.random.RandomState(seed)

    features_path = tmp_path / "features.csv"
    price_path = tmp_path / "price.csv"

    df_features = pd.DataFrame(
        rng.normal(size=(len(close_prices), 3)), index=dates, columns=["feat_a", "feat_b", "feat_c"]
    )
    df_features.index.name = "date"
    df_features.to_csv(features_path)

    df_price = pd.DataFrame({"Close": close_prices}, index=dates)
    df_price.index.name = "date"
    df_price.to_csv(price_path)

    return features_path, price_path, dates


# ─── classify_action_distribution ───────────────────────────────────────────


def test_classify_flags_always_flat_as_collapsed():
    actions = np.zeros(100)
    is_collapsed, label, stats = classify_action_distribution(actions)
    assert is_collapsed
    assert "hold-cash" in label


def test_classify_flags_always_max_long_as_collapsed():
    actions = np.ones(100)
    is_collapsed, label, stats = classify_action_distribution(actions)
    assert is_collapsed
    assert "max-long" in label


def test_classify_flags_always_max_short_as_collapsed():
    actions = -np.ones(100)
    is_collapsed, label, stats = classify_action_distribution(actions)
    assert is_collapsed
    assert "max-short" in label


def test_classify_does_not_flag_varied_actions():
    rng = np.random.RandomState(0)
    actions = rng.uniform(-1, 1, 200)
    is_collapsed, label, stats = classify_action_distribution(actions)
    assert not is_collapsed


# ─── helper functions ────────────────────────────────────────────────────────


def test_portfolio_values_to_returns():
    values = np.array([100.0, 110.0, 99.0])
    returns = portfolio_values_to_returns(values)
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.10)


def test_momentum_metrics_on_window(tmp_path):
    close_prices = 100 + np.cumsum(np.random.RandomState(1).normal(0, 1, N_DAYS))
    _, price_path, dates = _write_env_csvs(tmp_path, close_prices)
    metrics = momentum_metrics_on_window(price_path, dates[30], dates[-1], window=5)
    assert metrics is not None
    assert "Sharpe Ratio" in metrics


# ─── PPO wiring smoke test ───────────────────────────────────────────────────


def test_ppo_trains_and_rolls_out_without_error(tmp_path):
    close_prices = 100 + np.cumsum(np.random.RandomState(2).normal(0, 1, N_DAYS))
    features_path, price_path, dates = _write_env_csvs(tmp_path, close_prices, seed=2)

    train_end = dates[N_DAYS // 2]
    val_start = dates[N_DAYS // 2 + 1]

    train_env = make_env("TEST", features_path, price_path, end_date=train_end)
    val_env = make_env("TEST", features_path, price_path, start_date=val_start)

    vec_env = DummyVecEnv([lambda: Monitor(train_env)])
    model = PPO("MlpPolicy", vec_env, n_steps=32, batch_size=16, n_epochs=1, seed=0, verbose=0)

    model.learn(total_timesteps=64, callback=ValidationSharpeCallback(val_env, eval_freq=32))

    portfolio_values, actions = rollout_deterministic(model, val_env)
    assert np.isfinite(portfolio_values).all()
    assert np.isfinite(actions).all()
    assert len(actions) == len(val_env.df) - 1

    is_collapsed, label, stats = classify_action_distribution(actions)
    assert isinstance(is_collapsed, bool)  # just confirm it runs end-to-end on a real policy
