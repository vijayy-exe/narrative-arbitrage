"""Tests for src/agents/attention_encoder.py (Upgrade 2 -- Transformer state
encoder). Mirrors tests/agents/test_agents.py / test_distributional_critic.py's
conventions: shape/reshape correctness as pure unit tests, plus a real (tiny)
end-to-end PPO training smoke test.
"""
import numpy as np
import pandas as pd
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.attention_encoder import TemporalAttentionExtractor
from src.agents.train_ppo import ValidationSharpeCallback, make_env, rollout_deterministic
from src.env.trading_env import NarrativeTradingEnv
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


# ─── shape / reshape correctness ────────────────────────────────────────────


def test_forward_output_shape_matches_features_dim():
    window, n_feat, features_dim = 6, 4, 16
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window * n_feat + 3,), dtype=np.float32)
    extractor = TemporalAttentionExtractor(obs_space, lookback_window=window, n_raw_features=n_feat,
                                            d_model=8, n_heads=2, n_layers=1, features_dim=features_dim)
    batch = th.randn(5, window * n_feat + 3)
    out = extractor(batch)
    assert out.shape == (5, features_dim)
    assert th.isfinite(out).all()


def test_constructor_rejects_mismatched_observation_shape():
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(99,), dtype=np.float32)
    with pytest.raises(AssertionError):
        TemporalAttentionExtractor(obs_space, lookback_window=6, n_raw_features=4)


def test_output_depends_on_final_timestep_not_an_earlier_one():
    """Regression test for the reshape/indexing in forward(): changing ONLY
    the last timestep's raw features must change the output; changing ONLY
    an earlier timestep must also change it (full sequence is attended
    over), but the two perturbations should produce DIFFERENT outputs from
    each other -- if the reshape silently transposed window/feature axes,
    perturbing "the last timestep" would not land where this test expects.
    """
    th.manual_seed(0)
    window, n_feat = 5, 3
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window * n_feat + 3,), dtype=np.float32)
    extractor = TemporalAttentionExtractor(obs_space, lookback_window=window, n_raw_features=n_feat,
                                            d_model=8, n_heads=2, n_layers=1, features_dim=8, dropout=0.0)
    extractor.eval()

    base = th.zeros(1, window * n_feat + 3)
    perturb_last = base.clone()
    perturb_last[0, (window - 1) * n_feat: window * n_feat] = 1.0
    perturb_first = base.clone()
    perturb_first[0, 0:n_feat] = 1.0

    with th.no_grad():
        out_base = extractor(base)
        out_last = extractor(perturb_last)
        out_first = extractor(perturb_first)

    assert not th.allclose(out_base, out_last)
    assert not th.allclose(out_base, out_first)
    assert not th.allclose(out_last, out_first)


# ─── end-to-end PPO wiring ───────────────────────────────────────────────────


def test_ppo_with_transformer_extractor_trains_and_rolls_out_without_error(tmp_path):
    close_prices = 100 + np.cumsum(np.random.RandomState(11).normal(0, 1, N_DAYS))
    features_path, price_path, dates = _write_env_csvs(tmp_path, close_prices, seed=11)

    train_end = dates[N_DAYS // 2]
    val_start = dates[N_DAYS // 2 + 1]
    window = 5

    train_env = NarrativeTradingEnv("TEST", features_path, price_path, end_date=train_end,
                                     lookback_window=window)
    val_env = NarrativeTradingEnv("TEST", features_path, price_path, start_date=val_start,
                                   lookback_window=window)
    n_raw_features = train_env.num_features

    vec_env = DummyVecEnv([lambda: Monitor(train_env)])
    model = PPO(
        "MlpPolicy", vec_env,
        policy_kwargs=dict(
            features_extractor_class=TemporalAttentionExtractor,
            features_extractor_kwargs=dict(lookback_window=window, n_raw_features=n_raw_features,
                                            d_model=8, n_heads=2, n_layers=1, features_dim=16),
        ),
        n_steps=32, batch_size=16, n_epochs=1, seed=0, verbose=0,
    )
    model.learn(total_timesteps=64, callback=ValidationSharpeCallback(val_env, eval_freq=32))

    portfolio_values, actions = rollout_deterministic(model, val_env)
    assert np.isfinite(portfolio_values).all()
    assert np.isfinite(actions).all()

    is_collapsed, label, stats = classify_action_distribution(actions)
    assert isinstance(is_collapsed, bool)
