"""Upgrade 6 (scoped) of narrative-arbitrage-implementation-upgrade.md --
behavior-cloning pretraining before PPO fine-tuning. The full option (a)
from that upgrade's dev prompt, not the Decision Transformer alternative
(b) -- DT is a materially larger build (a from-scratch sequence model
trained on return-to-go-conditioned trajectories) that this scoped pass
does not attempt; BC pretraining directly tests the same underlying
hypothesis (does giving the policy a warm start, instead of pure
from-scratch online exploration, reduce the single-window-vs-walk-forward
instability this project has documented repeatedly -- e.g. DOGE: 3.95
single-window vs 1.71 walk-forward average) with a much smaller build.

MECHANISM: before PPO.learn() ever runs, the policy's action distribution
is pretrained with ordinary supervised regression (MSE) to mimic the
momentum baseline's actions (src/baselines/momentum.py's own rule,
reimplemented here read-only against the env's own price series -- not
reused via import, since momentum.py's public function operates on a full
CSV file path, not a already-constructed env's in-memory price series) on
the TRAIN window. The target is the Gaussian policy's MEAN action
(`get_distribution(obs).distribution.mean`, confirmed differentiable) --
regressing the mean toward a known-reasonable action gives PPO exploration
a starting point that isn't random-network noise, without literally
copying the momentum rule's logic into the RL loss (which remains PPO's
own, unmodified, once pretraining ends).
"""
import numpy as np
import torch as th
import torch.nn.functional as F


def momentum_action_sequence(env, window=20):
    """Recomputes src/baselines/momentum.py's exact rule
    (`position = 1 if trailing window-day return > 0 else -1`) against
    `env`'s own in-memory price series (`env.df['Close']`), aligned to
    env's own row order -- read-only, no env state is mutated. Returns an
    array of length len(env.df) (one action per row, matching every other
    per-step action array in this codebase)."""
    close = env.df["Close"]
    momentum = close.pct_change(window)
    positions = np.where(momentum > 0, 1.0, -1.0)
    return positions.astype(np.float32)


def pretrain_behavior_cloning(model, env, epochs=10, lr=1e-3, batch_size=32, seed=0):
    """Pretrains `model.policy`'s action distribution (in place) to mimic
    the momentum rule's actions on `env`'s full episode, via ordinary
    supervised MSE on the Gaussian policy's mean action. Steps `env`
    ONCE (using the momentum actions themselves, so observations reflect
    a momentum-following portfolio trajectory, not an untrained-policy
    one -- a coherent demonstration trajectory, not one contaminated by
    the policy's own pre-pretraining noise) to collect (obs, target_action)
    pairs, then trains for `epochs` passes over that fixed dataset.
    Returns the list of per-epoch mean losses (for logging/diagnostics).
    """
    rng = np.random.RandomState(seed)
    target_actions_full = momentum_action_sequence(env)

    obs_buffer = []
    action_buffer = []
    obs, _ = env.reset()
    step = 0
    done = False
    while not done:
        obs_buffer.append(obs)
        target = target_actions_full[min(step, len(target_actions_full) - 1)]
        if not np.isfinite(target):
            target = -1.0  # matches momentum.py's own NaN->short convention (see infer.py's docstring)
        action_buffer.append([target])
        obs, reward, done, truncated, info = env.step(np.array([target], dtype=np.float32))
        step += 1

    obs_arr = np.array(obs_buffer, dtype=np.float32)
    action_arr = np.array(action_buffer, dtype=np.float32)
    n = len(obs_arr)

    device = model.policy.device
    obs_t_full = th.as_tensor(obs_arr, device=device)
    action_t_full = th.as_tensor(action_arr, device=device)

    losses = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            obs_batch = obs_t_full[idx]
            action_batch = action_t_full[idx]

            dist = model.policy.get_distribution(obs_batch)
            predicted_mean = dist.distribution.mean
            loss = F.mse_loss(predicted_mean, action_batch)

            model.policy.optimizer.zero_grad()
            loss.backward()
            model.policy.optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(float(np.mean(epoch_losses)))

    return losses
