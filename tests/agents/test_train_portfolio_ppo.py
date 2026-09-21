"""Tests for src/agents/train_portfolio_ppo.py (Upgrade 5). Uses the REAL
repo data (data/processed/, data/raw/price/, models/ppo_<asset>/) since
this script's whole purpose is joint training across the real 5-asset
basket -- mirrors the convention tests/realtime/test_infer.py already
uses for real-checkpoint tests (no chdir, no synthetic fixtures for this
file specifically).
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.train_portfolio_ppo import (
    BASKET,
    get_common_train_val_split,
    independent_policies_portfolio_baseline,
    make_multi_asset_env,
    portfolio_values_to_returns,
    rollout_portfolio_deterministic,
)


def test_get_common_train_val_split_returns_ordered_dates():
    train_end, val_start, val_end = get_common_train_val_split()
    assert train_end < val_start <= val_end


def test_make_multi_asset_env_covers_full_basket():
    env = make_multi_asset_env()
    assert env.symbols == BASKET
    assert env.n_assets == 5


def test_independent_policies_baseline_is_finite_and_date_aligned():
    train_end, val_start, val_end = get_common_train_val_split()
    values = independent_policies_portfolio_baseline(val_start, val_end)
    assert np.isfinite(values).all()
    val_env = make_multi_asset_env(start_date=val_start, end_date=val_end)
    assert len(values) == len(val_env.dates)


def test_tiny_joint_ppo_trains_and_rolls_out_without_error():
    train_end, val_start, val_end = get_common_train_val_split()
    train_env = make_multi_asset_env(end_date=train_end)
    val_env = make_multi_asset_env(start_date=val_start, end_date=val_end)

    vec_env = DummyVecEnv([lambda: Monitor(train_env)])
    model = PPO("MlpPolicy", vec_env, n_steps=32, batch_size=16, n_epochs=1, seed=0, verbose=0)
    model.learn(total_timesteps=64)

    values = rollout_portfolio_deterministic(model, val_env)
    assert np.isfinite(values).all()
    returns = portfolio_values_to_returns(values)
    assert len(returns) == len(values) - 1
