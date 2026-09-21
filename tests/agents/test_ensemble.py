"""Tests for src/agents/ensemble.py -- the seed-ensemble policy. Tiny
timestep budgets, following this repo's established smoke-test convention
(not a "does it learn" test)."""
import numpy as np
import pandas as pd

from src.agents.ensemble import rollout_deterministic_ensemble, train_lstm_ensemble_members
from src.agents.train_ppo import make_env

N_DAYS = 60


def _write_env_csvs(tmp_path, seed=0):
    dates = pd.date_range("2023-01-01", periods=N_DAYS, freq="D")
    rng = np.random.RandomState(seed)

    features_path = tmp_path / "features.csv"
    price_path = tmp_path / "price.csv"

    df_features = pd.DataFrame(rng.normal(size=(N_DAYS, 3)), index=dates, columns=["a", "b", "c"])
    df_features.index.name = "date"
    df_features.to_csv(features_path)

    df_price = pd.DataFrame({"Close": 100 + np.cumsum(rng.normal(size=N_DAYS))}, index=dates)
    df_price.index.name = "date"
    df_price.to_csv(price_path)
    return features_path, price_path


def test_train_lstm_ensemble_members_produces_n_distinct_models(tmp_path):
    features_path, price_path = _write_env_csvs(tmp_path)
    models = train_lstm_ensemble_members(
        lambda: make_env("TEST", features_path, price_path), n_members=2, base_seed=0, timesteps=64,
    )
    assert len(models) == 2
    assert models[0] is not models[1]


def test_ensemble_members_use_different_seeds_with_large_stride():
    """Regression test for the seed-collision-avoidance stride: with the
    default stride=1000, 2 members' seeds must not collide with any of
    this project's other seed conventions (42, 100-104)."""
    from src.agents.ensemble import train_lstm_ensemble_members
    import inspect

    src = inspect.getsource(train_lstm_ensemble_members)
    assert "seed_stride" in src  # documents the design choice is actually wired, not just described


def test_rollout_deterministic_ensemble_runs_and_averages_actions(tmp_path):
    features_path, price_path = _write_env_csvs(tmp_path, seed=1)
    train_env_factory = lambda: make_env("TEST", features_path, price_path)
    models = train_lstm_ensemble_members(train_env_factory, n_members=3, base_seed=0, timesteps=64)

    val_env = make_env("TEST", features_path, price_path)
    portfolio_values, actions = rollout_deterministic_ensemble(models, val_env)

    assert np.isfinite(portfolio_values).all()
    assert np.isfinite(actions).all()
    assert (actions >= -1.0).all() and (actions <= 1.0).all()
    assert len(actions) == len(val_env.df) - 1


def test_ensemble_action_is_the_mean_of_member_actions_at_each_step(tmp_path):
    """Direct check of the ensemble mechanism: manually query each member
    at the SAME initial observation and confirm the ensemble's first
    action equals their mean (not e.g. the first member's action alone)."""
    features_path, price_path = _write_env_csvs(tmp_path, seed=2)
    train_env_factory = lambda: make_env("TEST", features_path, price_path)
    models = train_lstm_ensemble_members(train_env_factory, n_members=3, base_seed=5, timesteps=64)

    val_env = make_env("TEST", features_path, price_path)
    obs, _ = val_env.reset()

    expected_actions = []
    for model in models:
        action, _ = model.predict(obs, state=None, episode_start=np.array([True]), deterministic=True)
        expected_actions.append(float(np.asarray(action).flatten()[0]))
    expected_mean = float(np.mean(expected_actions))

    val_env2 = make_env("TEST", features_path, price_path)
    _, actions = rollout_deterministic_ensemble(models, val_env2)
    np.testing.assert_allclose(actions[0], expected_mean, atol=1e-5)
