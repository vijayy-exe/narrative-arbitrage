import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_pipeline.features import recompute_derivatives_on_index


def expanding_normalize(raw):
    """Causal (expanding-window) z-score normalization of a feature
    DataFrame: at row t, mean/std are computed only from rows <= t, so this
    never leaks future distributional statistics into an earlier timestep's
    observation (see __init__'s longer comment for why). Leading rows with
    no history yet (or any NaN raw feature, e.g. a live source that failed
    to fetch that day -- see src/realtime/infer.py) fall back to 0.0 (the
    running mean), which is a standard imputation convention, not a claim
    about the underlying signal.

    Factored out of __init__ (2026-09, Phase 7) so src/realtime/infer.py can
    build a live observation vector with the EXACT SAME normalization this
    env uses for training/backtesting, instead of reimplementing it.
    """
    expanding_mean = raw.expanding().mean()
    expanding_std = raw.expanding().std()
    return ((raw - expanding_mean) / (expanding_std + 1e-6)).fillna(0.0)


class NarrativeTradingEnv(gym.Env):
    """
    RL Environment for trading based on narrative attention signals.
    """
    def __init__(self, asset_symbol, features_path, price_path,
                 initial_cash=10000.0,
                 transaction_fee=0.001,
                 risk_lambda=0.1,
                 volatility_window=20,
                 start_date=None,
                 end_date=None,
                 lookback_window=1):
        super(NarrativeTradingEnv, self).__init__()

        self.asset_symbol = asset_symbol
        self.transaction_fee = transaction_fee
        self.risk_lambda = risk_lambda
        self.initial_cash = initial_cash
        # Upgrade 2 (narrative-arbitrage-implementation-upgrade.md, attention-
        # based encoder): lookback_window=1 (the default) is BYTE-FOR-BYTE
        # the original single-timestep observation -- every existing
        # checkpoint (models/ppo_*, models_cvar/ppo_*) was trained with this
        # default and remains valid/reproducible. lookback_window=N>1
        # widens _get_observation() to the last N timesteps of normalized
        # features (flattened), for src/agents/attention_encoder.py's
        # Transformer feature extractor to reshape back into a (window,
        # n_features) sequence -- see that module's docstring. The 3-scalar
        # portfolio context is NEVER windowed (it describes "right now", not
        # a history), so it's appended once regardless of lookback_window.
        assert lookback_window >= 1, f"lookback_window must be >= 1, got {lookback_window}"
        self.lookback_window = lookback_window

        # Load data
        self.features = pd.read_csv(features_path, index_col='date', parse_dates=True).sort_index()
        self.prices = pd.read_csv(price_path, index_col='date', parse_dates=True).sort_index()

        # Align features and prices
        full_df = self.features.join(self.prices[['Close']], how='inner').sort_index()

        # BUG FIX (found 2026-09-07): the joined features carry velocity/
        # acceleration columns computed by features.py on the full
        # CALENDAR-DAY grid (7 days/week). For stock assets (GME, AMC),
        # `self.prices` only has weekday rows, so inner-joining against it
        # silently discarded search_interest_velocity/_acceleration
        # entirely -- Google Trends' weekly reports are anchored to a fixed
        # day (Sunday) that a stock never trades on, so that signal was a
        # constant zero for GME/AMC in every episode despite being the
        # single strongest cross-platform predictor for the crypto assets
        # (which trade 7 days/week, so never hit this). Recomputing here,
        # on this env's own (price-joined) index, makes velocity mean
        # "change since this asset's last actual trading day" -- still
        # strictly causal (see recompute_derivatives_on_index()'s
        # docstring) and the economically correct question for a same-day
        # trading decision anyway.
        full_df = recompute_derivatives_on_index(full_df)

        self.current_step = 0

        self.feature_columns = [c for c in full_df.columns if c != 'Close']

        # Normalize observations (dev plan Phase 3: "normalize/scale all
        # inputs" -- feature magnitudes vary wildly, e.g. raw pageviews in
        # the thousands vs. a position in [-1, 1], which destabilizes PPO).
        # Uses an EXPANDING (causal) z-score, not a single global fit, for
        # the same reason src/data_pipeline/features.py normalizes
        # sentiment_divergence with an expanding min/max instead of a
        # whole-series one: a global fit computed once over the whole file
        # leaks future distributional statistics (mean/std shift once more
        # data arrives) into every earlier timestep's observation. An
        # expanding stat at time t depends only on rows <= t, so it stays
        # leak-free regardless of how this file is later sliced into
        # train/val/test windows.
        #
        # Computed over the FULL file BEFORE any start_date/end_date
        # windowing below: a validation window's agent would, in reality,
        # have access to all prior history (including the training period)
        # when deciding how to normalize "is this pageview count unusually
        # high" -- not just the handful of rows inside its own window.
        # Slicing first and expanding second would give a validation window
        # a cold start with almost no history to normalize against.
        # NOTE on the .fillna(0.0) below: src/data_pipeline/features.py
        # deliberately leaves velocity/acceleration columns as NaN for the
        # first 1-2 rows of every asset (a derivative genuinely has no value
        # on day 1 -- see docs/feature_dictionary.md) rather than fabricate
        # a fake "0 change". That's the right call for the processed CSV,
        # but an RL observation vector must still be a real number every
        # step. Post-normalization, filling with 0 means "no history yet for
        # this feature -> present it as network-neutral (at the running
        # mean)" to the policy, which is a standard imputation convention,
        # not a factual claim about the underlying signal (unlike zero-
        # filling the raw CSV would be). Verified empirically against real
        # fetched data (all 517 DOGE steps), not just assumed -- see
        # tests/env/test_env.py::test_leading_nan_features_do_not_reach_observation.
        raw = full_df[self.feature_columns]
        normalized_full = expanding_normalize(raw)

        # Rolling realized volatility of the asset's own daily % returns, in
        # RETURN units (not price units) -- used to scale the risk penalty
        # below so it's comparable across assets of wildly different price
        # levels (see step()'s reward comment for why this replaced an
        # earlier "position magnitude as a volatility proxy" version, which
        # was quantitatively wrong by ~100x on real data, not just a loose
        # approximation). Rolling, not expanding: a realized-volatility
        # *risk estimate* should track recent conditions, unlike the
        # feature-normalization stats above where "how unusual is this
        # value relative to all history" is the intended, deliberately
        # slow-moving question. Causal either way -- window only looks
        # backward from t. Cold-start rows (fewer than `volatility_window`
        # prior returns) fall back to 0 (no risk penalty) rather than a
        # fabricated volatility estimate from insufficient history.
        pct_returns = full_df['Close'].pct_change()
        volatility_full = pct_returns.rolling(volatility_window, min_periods=2).std().fillna(0.0)

        if start_date is not None or end_date is not None:
            mask = pd.Series(True, index=full_df.index)
            if start_date is not None:
                mask &= full_df.index >= pd.to_datetime(start_date)
            if end_date is not None:
                mask &= full_df.index <= pd.to_datetime(end_date)
            self.df = full_df.loc[mask]
            self._normalized_features = normalized_full.loc[mask]
            self._volatility = volatility_full.loc[mask]
        else:
            self.df = full_df
            self._normalized_features = normalized_full
            self._volatility = volatility_full

        if len(self.df) < 2:
            raise ValueError(
                f"{asset_symbol}: date window [{start_date}, {end_date}] leaves only "
                f"{len(self.df)} aligned rows -- need at least 2 to take a step."
            )

        # Action space: Target position size [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space: [lookback_window x Features] (flattened) + Portfolio Context
        # Portfolio Context: [position, cash, unrealized_pnl] -- never windowed.
        self.num_features = len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_features * self.lookback_window + 3,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash  # cash IS total portfolio value here -- see step()
        self.position = 0.0  # target position as a FRACTION of initial_cash, -1..1 (not a unit/share count)
        self.entry_price = None  # price when `self.position` was last set to a new nonzero value
        self.portfolio_value = self.initial_cash
        self.unrealized_pnl = 0.0

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        if self.lookback_window == 1:
            # Original path, unchanged -- kept as its own branch (rather than
            # letting the window==1 case fall through the general logic
            # below) so this exact code path, and therefore every existing
            # checkpoint trained against it, is untouched by the
            # lookback_window feature.
            feat = self._normalized_features.iloc[self.current_step].values.astype(np.float32)
        else:
            # Last `lookback_window` rows up to and including current_step,
            # LEFT-ZERO-PADDED at the start of the episode where fewer than
            # `lookback_window` rows of history exist yet. Zero is the same
            # "neutral / no information" fill expanding_normalize() already
            # uses for a feature's own missing history (see that function's
            # docstring), so padding whole rows with it is consistent with
            # the existing convention, not a new one.
            lo = self.current_step - self.lookback_window + 1
            window = self._normalized_features.iloc[max(0, lo):self.current_step + 1].values.astype(np.float32)
            if lo < 0:
                pad = np.zeros((-lo, self.num_features), dtype=np.float32)
                window = np.concatenate([pad, window], axis=0)
            feat = window.reshape(-1)  # flatten to (lookback_window * num_features,)

        # Portfolio context
        # Normalized position, normalized cash, normalized pnl
        context = np.array([
            self.position,
            self.cash / self.initial_cash,
            self.unrealized_pnl / self.initial_cash
        ], dtype=np.float32)

        return np.concatenate([feat, context])

    def step(self, action):
        # Action is the target position [-1, 1], a FRACTION of initial_cash
        # -- e.g. position=1.0 means "$initial_cash worth of notional long
        # exposure", regardless of the asset's own price level. This
        # replaced an earlier "position 1.0 = 100 units" scheme (fixed
        # 2026-09): a fixed unit count makes "full position" worth a wildly
        # different, arbitrary dollar amount depending on each asset's price
        # (100 units of a $0.15 DOGE vs a $20 GME is not a comparable bet
        # size at all) -- confirmed on real data: for DOGE this made a full
        # +-1 position's daily P&L ~100x smaller than the risk penalty
        # below, so the agent could only ever learn "stay near flat" no
        # matter how long it trained. Sizing by initial_cash instead makes
        # "how big is a full position" consistent and comparable across
        # every asset in the basket.
        target_position = float(np.clip(action[0], -1.0, 1.0))

        # 1. Get current price
        current_price = self.df.iloc[self.current_step]['Close']

        # 2. Transaction cost: fee * dollar notional traded (not fee * unit
        # count * price -- same price-scale-invariance fix as above).
        cost = abs(target_position - self.position) * self.initial_cash * self.transaction_fee

        # 3. Realized P&L from the OLD position over the price move that
        # already happened (current_step-1 -> current_step). The position
        # actually held during that move was self.position (pre-rebalance),
        # NOT target_position -- crediting target_position here would give
        # the agent reward for a price move that occurred before it took
        # that position (a look-ahead bug, fixed 2026-09: the previous
        # version computed step_return correctly here but then silently
        # discarded it and re-derived a *different*, wrong quantity for the
        # reward -- see dev-plan review notes for the old formula).
        if self.current_step > 0:
            prev_price = self.df.iloc[self.current_step - 1]['Close']
            pct_change = (current_price - prev_price) / prev_price
            step_return = self.position * pct_change * self.initial_cash
        else:
            step_return = 0.0

        # `self.cash` IS total portfolio value in this scheme: every step's
        # realized P&L and cost flow directly into it (dense, daily
        # mark-to-market), so there is no separate "position value" to add
        # on top afterwards -- doing so would double-count the same
        # notional exposure as both "cash on hand" and "money in the
        # position" simultaneously, since no principal is ever actually
        # debited from cash to "buy" the position under this notional-P&L
        # (CFD-style) accounting.
        self.cash += step_return - cost

        # Track entry price purely for the unrealized_pnl OBSERVATION
        # feature below -- this does not feed back into cash/reward (which
        # already fully mark-to-market daily above; a dense per-step reward
        # is standard and important for RL credit assignment, so it is
        # deliberately NOT deferred to "only when the position is closed").
        # Without this, unrealized_pnl would just be a constant multiple of
        # `self.position` (zero extra information for the policy); with it,
        # it answers a genuinely different question -- "is my current bet,
        # opened however many days ago, up or down right now" -- that a
        # single-timestep observation can't otherwise see.
        if target_position != self.position:
            self.entry_price = current_price if target_position != 0 else None
        self.position = target_position

        if self.position != 0 and self.entry_price is not None:
            self.unrealized_pnl = self.position * self.initial_cash * (current_price - self.entry_price) / self.entry_price
        else:
            self.unrealized_pnl = 0.0

        self.portfolio_value = self.cash

        # 4. Calculate Reward: realized return net of transaction cost,
        # minus a risk penalty on position size scaled by the asset's own
        # recent realized volatility. This is R_t = Return_t - lambda*Vol -
        # Cost(dPos) from docs/mdp_spec.md -- an earlier version used
        # position magnitude alone as a stand-in for "Vol" (no rolling
        # volatility was tracked), which turned out not to be a loose
        # approximation but a real bug: on real DOGE data the resulting
        # risk penalty was ~126x larger than the actual return signal for a
        # full position, meaning ANY training budget would converge toward
        # "always stay near flat" regardless of the actual attention
        # features, independent of the price-scale bug above. Multiplying
        # by self._volatility (rolling std of daily % returns, causal, see
        # __init__) puts both terms of the reward in the same return-scale
        # units, so risk_lambda becomes a genuinely tunable "how much do I
        # dislike volatility exposure relative to expected return"
        # coefficient instead of an arbitrary constant with no principled
        # scale.
        volatility_t = self._volatility.iloc[self.current_step]
        reward = (step_return - cost) / self.initial_cash
        reward -= self.risk_lambda * abs(self.position) * volatility_t

        # 5. Advance step
        self.current_step += 1

        done = self.current_step >= len(self.df) - 1
        truncated = False

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, float(reward), done, truncated, {"portfolio_value": self.portfolio_value}

    def render(self):
        print(f"Step: {self.current_step} | Position: {self.position:.2f} | Cash: {self.cash:.2f} | Value: {self.portfolio_value:.2f}")
