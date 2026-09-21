"""Tier 2, Phase 5 -- epistemic-uncertainty-aware position sizing.

Distinct from Upgrade 1's CVaR objective (which reshapes the *distribution*
of predicted returns the critic targets) and from src/agents/ensemble.py's
seed ensemble (which averages ACTIONS across members to reduce variance in
the decision itself): this targets a different, complementary question --
does the agent know when it doesn't know? A deep ensemble of independently
-initialized critics' DISAGREEMENT on a given state's value is a standard
epistemic-uncertainty proxy (high disagreement = states unlike anything
seen enough during training to have a confident value estimate). This
module scales the POSITION SIZE down when that disagreement is high, as a
post-hoc wrapper applicable to any already-trained policy without
retraining it -- distinct from ensemble.py, which trains a fresh ensemble
of ACTORS whose actions are averaged; here one policy's action is scaled
by an independently-trained ensemble of CRITICS' disagreement.
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def train_critic_ensemble(train_env_factory, n_members, base_seed, timesteps, seed_stride=1000):
    """Trains `n_members` independently-seeded, independently-initialized
    PPO models (full actor-critic, not just a bare value network -- SB3
    has no standalone-critic training path, so an ensemble of full models
    is the practical way to get an ensemble of critics) on independent
    copies of the training env. Mirrors src/agents/ensemble.py's seed-
    stride convention (large stride to avoid colliding with this
    project's other seed conventions: 42, 100-104)."""
    models = []
    for i in range(n_members):
        member_seed = base_seed + i * seed_stride
        env = train_env_factory()
        vec_env = DummyVecEnv([lambda env=env: Monitor(env)])
        model = PPO("MlpPolicy", vec_env, seed=member_seed, verbose=0)
        model.learn(total_timesteps=timesteps)
        models.append(model)
    return models


def value_disagreement(models, obs):
    """Std, across `models`' own critics, of their predicted value for
    `obs` -- the epistemic uncertainty signal. Uses each model's
    `predict_values` (SB3's own critic-inference extension point, the
    same one src/agents/distributional_critic.py's CVaR mechanism relies
    on elsewhere in this project) rather than reimplementing a forward
    pass through each policy's value network by hand."""
    import torch as th

    values = []
    for model in models:
        obs_t = th.as_tensor(obs, device=model.policy.device).float().unsqueeze(0)
        with th.no_grad():
            v = model.policy.predict_values(obs_t)
        values.append(float(v.item()))
    return float(np.std(values)), values


def scale_action_by_uncertainty(action, disagreement, k=2.0):
    """Scales `action` down as `disagreement` grows: multiplier =
    exp(-k * disagreement) -- smoothly in (0, 1], equal to 1.0 at zero
    disagreement (no scaling when the ensemble agrees) and decaying
    toward 0 as disagreement grows, with `k` controlling how aggressively.
    A simple, monotonic, boundedly-sane choice -- not claimed to be the
    only reasonable functional form, just a defensible one."""
    multiplier = float(np.exp(-k * disagreement))
    return np.asarray(action) * multiplier, multiplier


def rollout_uncertainty_aware(base_model, critic_ensemble, env, k=2.0):
    """Steps `env` using `base_model`'s own deterministic action, scaled by
    `critic_ensemble`'s value disagreement at each step (scale_action_by_
    uncertainty). `base_model` is untouched/not retrained -- this is the
    "post-hoc wrapper, testable against any existing trained policy"
    design from this module's docstring."""
    obs, info = env.reset()
    portfolio_values = [env.portfolio_value]
    actions = []
    disagreements = []
    done = False
    while not done:
        raw_action, _ = base_model.predict(obs, deterministic=True)
        disagreement, _ = value_disagreement(critic_ensemble, obs)
        scaled_action, _ = scale_action_by_uncertainty(raw_action, disagreement, k=k)
        actions.append(float(np.asarray(scaled_action).flatten()[0]))
        disagreements.append(disagreement)

        obs, reward, done, truncated, info = env.step(scaled_action)
        portfolio_values.append(info["portfolio_value"])
    return np.array(portfolio_values), np.array(actions), np.array(disagreements)
