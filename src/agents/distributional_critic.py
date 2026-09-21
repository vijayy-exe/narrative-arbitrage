"""Risk-sensitive (CVaR) PPO -- Upgrade 1 of `narrative-arbitrage-
implementation-upgrade.md`, in direct response to AMC/GME's catastrophic
single-window Sharpe (-2.03 / -3.82, see docs/CURRENT_STATUS.md) under
plain expected-return PPO. `docs/reward_hacking_findings.md` already ruled
out reward hacking/policy collapse as the cause for both, and diagnosed
GME's failure as a genuine directional miss (net short into a +72.6%
rally) and AMC's as likely overconfident sizing/overtrading -- this module
targets exactly that: a policy that is only ever told to maximize EXPECTED
return has no incentive to hedge against being confidently wrong.

MECHANISM (a QR-DQN/IQN-style quantile critic, adapted to on-policy PPO):

1. `QuantileCriticPolicy` replaces the standard scalar value head
   (`nn.Linear(latent_dim_vf, 1)`) with an `n_quantiles`-wide head that
   predicts `n_quantiles` evenly-spaced quantiles of the return
   distribution instead of a single expected value.

2. Wherever SB3 needs "the value" (forward(), evaluate_actions(),
   predict_values() -- these are SB3's own extension points, called by its
   UNMODIFIED rollout-collection and GAE code), this policy returns
   CVaR_alpha of the predicted quantile distribution (the mean of the
   bottom `alpha` fraction of quantiles) INSTEAD of the mean. This is the
   entire mechanism: GAE (stock SB3 code, not touched) then computes
   advantages relative to a pessimistic, tail-aware value baseline rather
   than the expected value, so an action that only looks good on average
   but carries a bad-tail risk gets a WORSE advantage than an action with
   the same mean but a safer tail -- exactly the asymmetry plain PPO
   cannot express. No changes to policy-loss/clipping/entropy are needed
   for this half of the mechanism.

3. Separately, `CVaRPPO` (a `PPO` subclass) replaces the critic's training
   signal: instead of MSE against the scalar GAE-lambda return target
   (`F.mse_loss(rollout_data.returns, values_pred)`, stock PPO.train()'s
   value loss), it fits the RAW quantile predictions to that same scalar
   target via the quantile Huber ("pinball") loss at each quantile's own
   fixed fraction tau_i = (i + 0.5) / n_quantiles. This is what actually
   makes the quantile head learn a real, differentiated distribution
   (rather than n_quantiles identical copies of the mean) -- CVaR-of-
   quantiles from step 2 would otherwise have nothing forcing separation
   between quantiles.

Everything else in CVaRPPO.train() (policy loss, clipping, entropy,
logging) is copied UNCHANGED from stable_baselines3.PPO.train() -- see
that method's source for the parts of this file that are deliberately
identical, not reimplemented from a different design.

SCOPE NOTE: this is a genuine, working distributional/CVaR-sensitive PPO,
not a full research-grade IQN (no implicit quantile *sampling* / learned
quantile embedding network -- n_quantiles is fixed at construction, not
sampled per forward pass). That simplification is deliberate: IQN's
learned quantile-embedding network is the main source of its added
complexity and is aimed at problems with much higher-dimensional return
distributions than a single scalar per-step PnL; a fixed QR-DQN-style grid
already gives a genuine tail-risk-aware critic for this problem's scale.
`clip_range_vf` (SB3's optional value-function clipping) is intentionally
NOT supported in CVaR mode -- clipping a scalar CVaR summary against old
CVaR values does not translate to a sound per-quantile clipping rule
without more design work than this ablation needs, so it is disabled with
an explicit warning rather than silently doing something unsound.
"""
import warnings
from functools import partial

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance


def quantile_huber_loss(quantiles: th.Tensor, target: th.Tensor, tau: th.Tensor, kappa: float = 1.0) -> th.Tensor:
    """Standard QR-DQN/IQN quantile Huber ("pinball-Huber") loss.

    quantiles: (batch, n_quantiles) predicted quantile values.
    target: (batch,) or (batch, 1) scalar regression target (here: PPO's
        GAE-lambda return) -- broadcast across every quantile, since PPO's
        return target is itself already a single scalar per timestep, not
        a set of Bellman-sampled next-state quantiles the way QR-DQN uses.
        This is quantile regression toward a scalar target: each quantile
        head learns to predict the tau_i-quantile of the SAMPLING
        distribution of that scalar target across the batch/minibatch,
        conditioned on the state -- a well-defined, standard form of
        distributional regression.
    tau: (n_quantiles,) the fixed quantile fractions, (i + 0.5)/n_quantiles.
    """
    if target.dim() == 1:
        target = target.unsqueeze(-1)
    td_error = target - quantiles  # (batch, n_quantiles)
    huber = th.where(
        td_error.abs() <= kappa,
        0.5 * td_error.pow(2),
        kappa * (td_error.abs() - 0.5 * kappa),
    )
    weight = th.abs(tau.unsqueeze(0) - (td_error.detach() < 0).float())
    return (weight * huber / kappa).mean()


class QuantileCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with a quantile-regression critic. See module
    docstring for the full mechanism. Drop-in for SB3's "MlpPolicy" string
    (pass the class itself, not a string, to PPO/CVaRPPO)."""

    def __init__(self, *args, n_quantiles: int = 32, cvar_alpha: float = 0.2, **kwargs):
        assert 0.0 < cvar_alpha <= 1.0, f"cvar_alpha must be in (0, 1], got {cvar_alpha}"
        self.n_quantiles = n_quantiles
        self.cvar_alpha = cvar_alpha
        super().__init__(*args, **kwargs)
        tau = (th.arange(n_quantiles, dtype=th.float32) + 0.5) / n_quantiles
        self.register_buffer("tau", tau)

    def _build(self, lr_schedule) -> None:
        # Build everything the normal way first (this also constructs the
        # optimizer over the STOCK scalar value_net's parameters) ...
        super()._build(lr_schedule)
        # ... then swap the scalar value head for a quantile head and
        # rebuild the optimizer so the new head's parameters are actually
        # included -- SB3 constructs self.optimizer from self.parameters()
        # at the END of _build(), i.e. BEFORE this swap runs, so skipping
        # this rebuild would silently leave the quantile head untrained
        # (no gradient ever reaching it).
        latent_dim_vf = self.mlp_extractor.latent_dim_vf
        self.value_net = nn.Linear(latent_dim_vf, self.n_quantiles)
        if self.ortho_init:
            self.value_net.apply(partial(self.init_weights, gain=1))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _quantiles_to_cvar(self, quantiles: th.Tensor) -> th.Tensor:
        """CVaR_alpha = mean of the bottom `alpha` fraction of predicted
        quantiles -- the pessimistic scalar "value" every SB3 extension
        point below returns instead of the mean. No monotonicity is
        enforced on the raw quantile head's outputs (a known, accepted
        QR-DQN-style simplification), so this sorts first."""
        sorted_q, _ = th.sort(quantiles, dim=-1)
        k = max(1, int(round(self.cvar_alpha * self.n_quantiles)))
        return sorted_q[:, :k].mean(dim=-1, keepdim=True)

    def predict_quantiles(self, obs) -> th.Tensor:
        """Raw (batch, n_quantiles) predicted return quantiles -- consumed
        by CVaRPPO.train()'s pinball-loss value-training step. Kept
        separate from forward()/evaluate_actions()/predict_values() (which
        must return the CVaR scalar for SB3's rollout/GAE code) rather than
        smuggled through as a side-channel attribute."""
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def forward(self, obs, deterministic: bool = False):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        quantiles = self.value_net(latent_vf)
        values = self._quantiles_to_cvar(quantiles)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        quantiles = self.value_net(latent_vf)
        values = self._quantiles_to_cvar(quantiles)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs):
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        quantiles = self.value_net(latent_vf)
        return self._quantiles_to_cvar(quantiles)


class CVaRPPO(PPO):
    """PPO with the value loss replaced by quantile-Huber regression
    against `QuantileCriticPolicy`'s raw quantile predictions. Requires
    `policy=QuantileCriticPolicy` (a class, not the "MlpPolicy" string) --
    raises if given anything else, rather than silently behaving like
    stock PPO with a wasted quantile head nobody trains correctly.

    Everything outside the value-loss block of train() is copied verbatim
    from stable_baselines3.PPO.train() -- see that method's source (pinned
    version: stable-baselines3==2.9.0, this repo's installed version) for
    what "unchanged" means here precisely.
    """

    def __init__(self, policy, env, *args, quantile_huber_kappa: float = 1.0, **kwargs):
        if kwargs.get("clip_range_vf") is not None:
            warnings.warn(
                "CVaRPPO does not support clip_range_vf (value-function clipping) -- "
                "see distributional_critic.py's module docstring for why. Ignoring it.",
                UserWarning,
            )
            kwargs = {**kwargs, "clip_range_vf": None}
        self.quantile_huber_kappa = quantile_huber_kappa
        super().__init__(policy, env, *args, **kwargs)
        if not isinstance(self.policy, QuantileCriticPolicy):
            raise TypeError(
                f"CVaRPPO requires policy=QuantileCriticPolicy (got {type(self.policy).__name__}) -- "
                "pass the class, not the 'MlpPolicy' string."
            )

    def train(self) -> None:
        import torch.nn.functional as F  # noqa: N812 -- matches SB3's own import alias

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # --- value loss: quantile-Huber regression, NOT MSE ---
                # (this is the one substantive deviation from PPO.train();
                # everything else in this method is copied verbatim)
                quantiles = self.policy.predict_quantiles(rollout_data.observations)
                value_loss = quantile_huber_loss(
                    quantiles, rollout_data.returns, self.policy.tau, kappa=self.quantile_huber_kappa
                )
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/cvar_alpha", self.policy.cvar_alpha)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
