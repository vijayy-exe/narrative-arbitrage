"""Seed-ensemble policy -- the direct answer to this project's single
biggest finding: `docs/multiseed_report.md` and
`docs/walk_forward_upgrade_validation.md`'s seed-robustness check both
found Sharpe varying almost as much as its own mean across identical-
hyperparameter seeds (GME/LSTM walk-forward: 1.38, 1.92, −0.62 across 3
seeds). A single trained policy is a high-variance point estimate of
"what this architecture can do" -- an ensemble of N independently-seeded
policies, whose actions are averaged at every step, is the standard,
well-motivated RL technique for trading policy variance for reliability.
This module builds and evaluates that ensemble for the LSTM architecture
(docs/architecture_ablation.md's basket-average and GME-specific winner),
the natural architecture to ensemble first.

MECHANISM: this is a genuine step-by-step ensemble POLICY, not a
post-hoc average of N separately-run backtests. At every timestep, each
member model is queried for its action (each threading its OWN LSTM
hidden state across the episode -- see rollout_deterministic_ensemble),
the actions are averaged, and THAT averaged action is what actually gets
stepped through the environment -- so each member's future observations
(portfolio state, price) reflect the ENSEMBLE's trajectory, not its own
individual one. This is a meaningfully different (and more realistic)
object than averaging N independent backtests' Sharpe ratios after the
fact.
"""
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def train_lstm_ensemble_members(train_env_factory, n_members, base_seed, timesteps,
                                 seed_stride=1000):
    """Trains `n_members` RecurrentPPO models on independently-constructed
    (but identically-configured) copies of the training env, one per
    member, seeded `base_seed + i * seed_stride` (a large stride so
    ensemble-member seeds never collide with this project's other seed
    conventions, e.g. 42, 100-104). `train_env_factory` is a zero-arg
    callable (not a shared env instance) so each member trains on its own
    independent env object, matching how every other trainer in this repo
    is called."""
    models = []
    for i in range(n_members):
        member_seed = base_seed + i * seed_stride
        env = train_env_factory()
        vec_env = DummyVecEnv([lambda env=env: Monitor(env)])
        model = RecurrentPPO("MlpLstmPolicy", vec_env, seed=member_seed, verbose=0)
        model.learn(total_timesteps=timesteps)
        models.append(model)
    return models


def rollout_deterministic_ensemble(models, env):
    """Steps `env` ONCE, taking the mean of all `models`' actions at each
    timestep as the actual action applied -- see module docstring for why
    this differs from averaging N separate backtests. Each model's own
    LSTM hidden state is threaded independently (a list of (state,
    episode_start) pairs, mirroring rollout_deterministic_recurrent's
    single-model version in architecture_ablation.py) since each member
    still needs its own recurrent memory of the shared observation
    sequence, even though they all see the same actions/rewards from here
    on (the ensemble's, not their own individual policy's)."""
    obs, info = env.reset()
    portfolio_values = [env.portfolio_value]
    actions = []
    states = [None] * len(models)
    episode_start = np.array([True])
    done = False
    while not done:
        member_actions = []
        for i, model in enumerate(models):
            action, states[i] = model.predict(obs, state=states[i], episode_start=episode_start,
                                                deterministic=True)
            member_actions.append(float(np.asarray(action).flatten()[0]))
        ensemble_action = np.array([np.mean(member_actions)], dtype=np.float32)
        actions.append(float(ensemble_action[0]))

        obs, reward, done, truncated, info = env.step(ensemble_action)
        episode_start = np.array([False])
        portfolio_values.append(info["portfolio_value"])
    return np.array(portfolio_values), np.array(actions)
