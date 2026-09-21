"""Tests for src/agents/behavior_cloning.py (Upgrade 6, scoped: BC
pretraining, not a full Decision Transformer -- see module docstring)."""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.behavior_cloning import momentum_action_sequence, pretrain_behavior_cloning
from src.agents.train_ppo import make_env

N_DAYS = 60


def _write_env_csvs(tmp_path, seed=0):
    dates = pd.date_range("2023-01-01", periods=N_DAYS, freq="D")
    rng = np.random.RandomState(seed)
    features_path = tmp_path / "features.csv"
    price_path = tmp_path / "price.csv"

    pd.DataFrame(rng.normal(size=(N_DAYS, 3)), index=dates, columns=["a", "b", "c"]) \
        .rename_axis("date").to_csv(features_path)
    pd.DataFrame({"Close": 100 + np.cumsum(rng.normal(size=N_DAYS))}, index=dates) \
        .rename_axis("date").to_csv(price_path)
    return features_path, price_path


def test_momentum_action_sequence_matches_baseline_sign_rule(tmp_path):
    features_path, price_path = _write_env_csvs(tmp_path)
    env = make_env("TEST", features_path, price_path)
    actions = momentum_action_sequence(env, window=20)
    assert len(actions) == len(env.df)
    assert set(np.unique(actions)) <= {-1.0, 1.0}

    close = env.df["Close"]
    momentum = close.pct_change(20)
    expected = np.where(momentum > 0, 1.0, -1.0)
    np.testing.assert_array_equal(actions, expected)


def test_pretrain_behavior_cloning_reduces_loss_over_epochs(tmp_path):
    features_path, price_path = _write_env_csvs(tmp_path, seed=1)
    env = make_env("TEST", features_path, price_path)
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    model = PPO("MlpPolicy", vec_env, seed=0, verbose=0)

    losses = pretrain_behavior_cloning(model, env, epochs=5, lr=1e-2, batch_size=8, seed=0)
    assert len(losses) == 5
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]  # loss should decrease as the policy learns to mimic momentum


def test_pretrained_policy_still_trains_normally_with_ppo_afterward(tmp_path):
    """The whole point: pretraining must leave the policy in a state PPO's
    own .learn() can continue training from normally, not a broken one."""
    features_path, price_path = _write_env_csvs(tmp_path, seed=2)
    env = make_env("TEST", features_path, price_path)
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    model = PPO("MlpPolicy", vec_env, n_steps=32, batch_size=16, n_epochs=1, seed=0, verbose=0)

    pretrain_behavior_cloning(model, env, epochs=3, lr=1e-2, batch_size=8, seed=0)
    model.learn(total_timesteps=64)  # must not raise

    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert np.isfinite(action).all()
