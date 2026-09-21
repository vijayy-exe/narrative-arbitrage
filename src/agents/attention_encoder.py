"""Attention-based (Transformer) state encoder -- Upgrade 2 of
`narrative-arbitrage-implementation-upgrade.md`.

The project's whole premise is trading on "attention dynamics" (velocity/
acceleration of Wikipedia/Trends/HN signals), but Upgrades 0-1 still fed
the policy a single flat vector per timestep -- no architecture in the loop
actually models how an acceleration PATTERN unfolds over several days, only
its instantaneous value. `TemporalAttentionExtractor` fixes that: it
consumes `NarrativeTradingEnv`'s `lookback_window`-flattened observation
(see that module's docstring), reshapes it back into a
(lookback_window, n_features) sequence, and runs a small Transformer
encoder over it, so the policy can learn e.g. "three consecutive days of
accelerating search-interest velocity" as a pattern, not just "today's
search-interest velocity is high."

This is registered as an SB3 `BaseFeaturesExtractor` -- SB3's own,
first-class extension point for swapping in a custom observation encoder
ahead of the standard actor/critic MLP heads. Unlike Upgrade 1's CVaR
critic, this needs NO changes to PPO's train()/GAE/rollout code at all:
`features_extractor_class=TemporalAttentionExtractor` in `policy_kwargs`
is the entire integration surface, by design.
"""
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TemporalAttentionExtractor(BaseFeaturesExtractor):
    """Reshapes a flattened (lookback_window * n_features + 3) observation
    back into (batch, lookback_window, n_features) [+ the 3 portfolio-
    context scalars, kept separate and un-attended, exactly as
    NarrativeTradingEnv itself never windows them -- see that module's
    docstring for why context describes "right now", not history],
    projects each timestep to `d_model`, adds a learned positional
    embedding (there IS a meaningful order here -- "3 days ago" vs
    "today" -- unlike e.g. a bag-of-tokens problem), and runs a small
    pre-norm Transformer encoder over the sequence. The final timestep's
    output embedding (not a mean-pool) is used as the sequence summary,
    concatenated with the 3 context scalars, and passed on as this
    extractor's `features_dim`-sized output -- SB3's standard MLP
    actor/critic heads sit on top of that, unchanged.

    Design choices, briefly:
    - Final-timestep (not mean-pool) summary: mirrors an autoregressive/
      "what does today look like given the recent pattern" framing, which
      matches this problem better than an order-agnostic pooled summary.
    - 2-4 layers, 4 heads, small d_model (default 32): this problem's
      lookback window is O(10-20) timesteps with O(20-30) raw features --
      far smaller than the NLP-scale sequences Transformers were designed
      for. A small model is a deliberate fit to the data scale, not a
      shortcut; a large Transformer would almost certainly overfit this
      little data (see docs/CURRENT_STATUS.md's existing single-window
      vs walk-forward instability finding -- more capacity would likely
      make that worse, not better).
    """

    def __init__(self, observation_space, lookback_window, n_raw_features,
                 d_model=32, n_heads=4, n_layers=2, features_dim=64, dropout=0.1):
        super().__init__(observation_space, features_dim=features_dim)
        assert observation_space.shape[0] == lookback_window * n_raw_features + 3, (
            f"TemporalAttentionExtractor expects a flattened "
            f"(lookback_window * n_raw_features + 3) observation, got shape "
            f"{observation_space.shape} for lookback_window={lookback_window}, "
            f"n_raw_features={n_raw_features} (expected "
            f"{lookback_window * n_raw_features + 3})."
        )
        self.lookback_window = lookback_window
        self.n_raw_features = n_raw_features

        self.input_proj = nn.Linear(n_raw_features, d_model)
        self.pos_embedding = nn.Parameter(th.zeros(1, lookback_window, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model + 3, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        seq_len = self.lookback_window * self.n_raw_features
        seq = observations[:, :seq_len].reshape(batch_size, self.lookback_window, self.n_raw_features)
        context = observations[:, seq_len:]  # (batch, 3) portfolio context, un-attended

        x = self.input_proj(seq) + self.pos_embedding
        x = self.transformer(x)
        summary = x[:, -1, :]  # final-timestep embedding, see class docstring

        return self.output_proj(th.cat([summary, context], dim=-1))
