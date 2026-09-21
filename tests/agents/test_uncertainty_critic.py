"""Tests for src/agents/uncertainty_critic.py (Tier 2, Phase 5)."""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.train_ppo import make_env
from src.agents.uncertainty_critic import (
    rollout_uncertainty_aware,
    scale_action_by_uncertainty,
    train_critic_ensemble,
    value_disagreement,
)

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


def test_scale_action_by_uncertainty_is_identity_at_zero_disagreement():
    action = np.array([0.7])
    scaled, multiplier = scale_action_by_uncertainty(action, disagreement=0.0, k=2.0)
    assert multiplier == 1.0
    np.testing.assert_allclose(scaled, action)


def test_scale_action_by_uncertainty_shrinks_toward_zero_as_disagreement_grows():
    action = np.array([0.7])
    _, m_low = scale_action_by_uncertainty(action, disagreement=0.1, k=2.0)
    _, m_high = scale_action_by_uncertainty(action, disagreement=5.0, k=2.0)
    assert 0.0 < m_high < m_low < 1.0


def test_value_disagreement_is_zero_for_identical_models(tmp_path):
    features_path, price_path = _write_env_csvs(tmp_path)
    env = make_env("TEST", features_path, price_path)
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    model = PPO("MlpPolicy", vec_env, seed=0, verbose=0)

    obs, _ = env.reset()
    disagreement, values = value_disagreement([model, model, model], obs)
    assert disagreement == 0.0
    assert len(set(values)) == 1


def test_train_critic_ensemble_and_uncertainty_rollout_runs_end_to_end(tmp_path):
    features_path, price_path = _write_env_csvs(tmp_path, seed=1)
    train_env_factory = lambda: make_env("TEST", features_path, price_path)

    critic_ensemble = train_critic_ensemble(train_env_factory, n_members=3, base_seed=0, timesteps=64)
    assert len(critic_ensemble) == 3

    base_env = train_env_factory()
    base_vec_env = DummyVecEnv([lambda: Monitor(base_env)])
    base_model = PPO("MlpPolicy", base_vec_env, seed=5, verbose=0)
    base_model.learn(total_timesteps=64)

    val_env = make_env("TEST", features_path, price_path)
    portfolio_values, actions, disagreements = rollout_uncertainty_aware(
        base_model, critic_ensemble, val_env, k=2.0
    )
    assert np.isfinite(portfolio_values).all()
    assert np.isfinite(actions).all()
    assert (disagreements >= 0).all()
    assert (np.abs(actions) <= 1.0).all()
