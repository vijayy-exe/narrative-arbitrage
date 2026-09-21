"""Tests for src/agents/distributional_critic.py (Upgrade 1 -- CVaR-aware
PPO). Mirrors tests/agents/test_agents.py's conventions: a couple of pure-
function/shape unit tests, plus a real (deliberately tiny) end-to-end
training smoke test against NarrativeTradingEnv, not a "does it learn"
performance test.
"""
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.distributional_critic import CVaRPPO, QuantileCriticPolicy, quantile_huber_loss
from src.agents.train_ppo import ValidationSharpeCallback, make_env, rollout_deterministic
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


# ─── quantile_huber_loss (pure function) ────────────────────────────────────


def test_quantile_huber_loss_decreases_as_quantiles_move_toward_target():
    th.manual_seed(0)
    n_quantiles = 16
    tau = (th.arange(n_quantiles, dtype=th.float32) + 0.5) / n_quantiles
    target = th.full((8,), 5.0)

    far = th.zeros(8, n_quantiles, requires_grad=False)
    close = th.full((8, n_quantiles), 4.9)

    loss_far = quantile_huber_loss(far, target, tau)
    loss_close = quantile_huber_loss(close, target, tau)
    assert loss_close.item() < loss_far.item()


def test_quantile_huber_loss_is_differentiable_and_reduces_loss_via_gradient_step():
    th.manual_seed(0)
    n_quantiles = 8
    tau = (th.arange(n_quantiles, dtype=th.float32) + 0.5) / n_quantiles
    quantiles = th.zeros(4, n_quantiles, requires_grad=True)
    target = th.full((4,), 3.0)

    loss_before = quantile_huber_loss(quantiles, target, tau)
    loss_before.backward()
    with th.no_grad():
        quantiles -= 0.5 * quantiles.grad
    quantiles.grad = None
    loss_after = quantile_huber_loss(quantiles.detach(), target, tau)

    assert loss_after.item() < loss_before.item()


# ─── QuantileCriticPolicy / CVaRPPO wiring ──────────────────────────────────


def test_ppo_with_quantile_policy_trains_and_rolls_out_without_error(tmp_path):
    close_prices = 100 + np.cumsum(np.random.RandomState(3).normal(0, 1, N_DAYS))
    features_path, price_path, dates = _write_env_csvs(tmp_path, close_prices, seed=3)

    train_end = dates[N_DAYS // 2]
    val_start = dates[N_DAYS // 2 + 1]

    train_env = make_env("TEST", features_path, price_path, end_date=train_end)
    val_env = make_env("TEST", features_path, price_path, start_date=val_start)

    vec_env = DummyVecEnv([lambda: Monitor(train_env)])
    model = CVaRPPO(
        QuantileCriticPolicy, vec_env,
        policy_kwargs=dict(n_quantiles=8, cvar_alpha=0.2),
        n_steps=32, batch_size=16, n_epochs=1, seed=0, verbose=0,
    )

    model.learn(total_timesteps=64, callback=ValidationSharpeCallback(val_env, eval_freq=32))

    portfolio_values, actions = rollout_deterministic(model, val_env)
    assert np.isfinite(portfolio_values).all()
    assert np.isfinite(actions).all()
    assert len(actions) == len(val_env.df) - 1

    is_collapsed, label, stats = classify_action_distribution(actions)
    assert isinstance(is_collapsed, bool)  # just confirm it runs end-to-end on a real policy


def test_quantile_head_parameters_are_included_in_optimizer(tmp_path):
    """Regression test for the _build() optimizer-rebuild step: if the
    quantile head's params were left out of the optimizer, this would pass
    silently (no error) but the critic would never actually learn."""
    close_prices = 100 + np.cumsum(np.random.RandomState(4).normal(0, 1, N_DAYS))
    features_path, price_path, dates = _write_env_csvs(tmp_path, close_prices, seed=4)
    env = make_env("TEST", features_path, price_path)
    vec_env = DummyVecEnv([lambda: Monitor(env)])

    model = CVaRPPO(QuantileCriticPolicy, vec_env, policy_kwargs=dict(n_quantiles=8), n_steps=32,
                     batch_size=16, n_epochs=1, seed=0, verbose=0)

    value_net_params = {id(p) for p in model.policy.value_net.parameters()}
    optimizer_params = {id(p) for group in model.policy.optimizer.param_groups for p in group["params"]}
    assert value_net_params.issubset(optimizer_params)
    assert model.policy.value_net.out_features == 8


def test_cvar_ppo_rejects_non_quantile_policy(tmp_path):
    close_prices = 100 + np.cumsum(np.random.RandomState(5).normal(0, 1, N_DAYS))
    features_path, price_path, dates = _write_env_csvs(tmp_path, close_prices, seed=5)
    env = make_env("TEST", features_path, price_path)
    vec_env = DummyVecEnv([lambda: Monitor(env)])

    try:
        CVaRPPO("MlpPolicy", vec_env, n_steps=32, batch_size=16, n_epochs=1, seed=0, verbose=0)
        assert False, "expected TypeError for a non-QuantileCriticPolicy"
    except TypeError as e:
        assert "QuantileCriticPolicy" in str(e)


def test_cvar_is_less_than_or_equal_to_mean_of_quantiles(tmp_path):
    """CVaR_alpha (mean of the bottom alpha-fraction of quantiles) must
    never exceed the plain mean of all quantiles -- this is what makes the
    resulting value baseline "pessimistic" rather than just a relabeled
    mean. True by construction (sorted, only bottom-k averaged) but worth
    locking down as an explicit regression test."""
    close_prices = 100 + np.cumsum(np.random.RandomState(6).normal(0, 1, N_DAYS))
    features_path, price_path, dates = _write_env_csvs(tmp_path, close_prices, seed=6)
    env = make_env("TEST", features_path, price_path)
    vec_env = DummyVecEnv([lambda: Monitor(env)])

    model = CVaRPPO(QuantileCriticPolicy, vec_env, policy_kwargs=dict(n_quantiles=32, cvar_alpha=0.2),
                     n_steps=32, batch_size=16, n_epochs=1, seed=0, verbose=0)

    obs = np.stack([vec_env.reset()[0] for _ in range(4)]).astype(np.float32)
    obs_t = th.as_tensor(obs)
    quantiles = model.policy.predict_quantiles(obs_t)
    cvar = model.policy._quantiles_to_cvar(quantiles)

    assert (cvar.squeeze(-1) <= quantiles.mean(dim=-1) + 1e-5).all()
