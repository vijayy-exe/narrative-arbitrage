"""Upgrade 5 of narrative-arbitrage-implementation-upgrade.md -- joint
multi-asset portfolio policy. Every architecture/objective evaluated so
far in this project (plain PPO, CVaR, LSTM, Transformer) is still five
INDEPENDENT single-asset policies with no way to express "DOGE and SHIB's
attention is spiking together, reduce size in both because they're
correlated" -- a real limitation named as future work in
`docs/final_report.md` section 8 until this module. `MultiAssetTradingEnv`
is a single environment whose observation/action span the whole basket
jointly, so ONE policy can learn genuine cross-asset behavior.

DATE ALIGNMENT: restricted to the INTERSECTION of trading days across
ALL assets in the basket (not each asset's own individual range like
NarrativeTradingEnv) -- a joint decision needs every asset's state on the
same calendar day, so a day missing for even one asset (e.g. a weekend,
which stocks don't trade but crypto does) is dropped for every asset, not
just the one missing it. For the default 5-asset basket (DOGE, GME, AMC,
PEPE, SHIB) this intersection is 270 trading days (2023-05-05 ->
2024-05-31), bounded by GME/AMC's weekday-only stock calendar and PEPE's
later start (no dedicated Wikipedia article history before then -- see
configs/assets.yaml's PEPE note). This is a real, deliberate cost of
joint training (270 days vs. e.g. DOGE's own 518-day individual range)
-- explicitly traded off in exchange for the cross-asset structure a
single-asset env cannot express at all.

ACTION SPACE: Box(-1, 1, shape=(N,)) -- one target position per asset,
each independently sized as a fraction of `initial_cash` (same convention
as NarrativeTradingEnv, NOT divided by N) -- i.e. leverage is not
explicitly capped when multiple assets are near their max simultaneously,
matching the single-asset env's own convention rather than introducing a
new, untested risk-scaling rule under this module's time budget. This
keeps the reward/portfolio-accounting formulas a direct, auditable sum
over per-asset formulas already validated in NarrativeTradingEnv, rather
than a new design.
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.data_pipeline.features import recompute_derivatives_on_index
from src.env.trading_env import expanding_normalize


class MultiAssetTradingEnv(gym.Env):
    def __init__(self, asset_paths, initial_cash=10000.0, transaction_fee=0.001,
                 risk_lambda=0.1, volatility_window=20, start_date=None, end_date=None):
        """asset_paths: dict {symbol: (features_path, price_path)}, in the
        fixed iteration order used for both the observation and action
        vectors (Python dicts preserve insertion order) -- callers must
        pass the same order every time they construct this env against a
        given trained policy."""
        super().__init__()
        self.symbols = list(asset_paths.keys())
        self.n_assets = len(self.symbols)
        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee
        self.risk_lambda = risk_lambda

        per_asset_normalized = {}
        per_asset_close = {}
        per_asset_volatility = {}
        per_asset_feature_cols = {}
        joined_indices = []

        for symbol, (features_path, price_path) in asset_paths.items():
            features = pd.read_csv(features_path, index_col="date", parse_dates=True).sort_index()
            prices = pd.read_csv(price_path, index_col="date", parse_dates=True).sort_index()
            full_df = features.join(prices[["Close"]], how="inner").sort_index()
            full_df = recompute_derivatives_on_index(full_df)  # see NarrativeTradingEnv for why

            feature_cols = [c for c in full_df.columns if c != "Close"]
            per_asset_feature_cols[symbol] = feature_cols
            per_asset_normalized[symbol] = expanding_normalize(full_df[feature_cols])
            per_asset_close[symbol] = full_df["Close"]

            pct_returns = full_df["Close"].pct_change()
            per_asset_volatility[symbol] = pct_returns.rolling(volatility_window, min_periods=2).std().fillna(0.0)

            joined_indices.append(full_df.index)

        common_index = joined_indices[0]
        for idx in joined_indices[1:]:
            common_index = common_index.intersection(idx)
        common_index = common_index.sort_values()

        if start_date is not None or end_date is not None:
            mask = pd.Series(True, index=common_index)
            if start_date is not None:
                mask &= common_index >= pd.to_datetime(start_date)
            if end_date is not None:
                mask &= common_index <= pd.to_datetime(end_date)
            common_index = common_index[mask]

        if len(common_index) < 2:
            raise ValueError(
                f"MultiAssetTradingEnv: only {len(common_index)} common trading days across "
                f"{self.symbols} in [{start_date}, {end_date}] -- need at least 2."
            )

        self.dates = common_index
        self.n_features_per_asset = [len(per_asset_feature_cols[s]) for s in self.symbols]
        self._normalized_features = {s: per_asset_normalized[s].loc[common_index] for s in self.symbols}
        self._close = {s: per_asset_close[s].loc[common_index] for s in self.symbols}
        self._volatility = {s: per_asset_volatility[s].loc[common_index] for s in self.symbols}

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        obs_dim = sum(self.n_features_per_asset) + 2 * self.n_assets + 1  # + [position_i, unrealized_pnl_i]*N + cash
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.positions = {s: 0.0 for s in self.symbols}
        self.entry_prices = {s: None for s in self.symbols}
        self.unrealized_pnl = {s: 0.0 for s in self.symbols}
        self.portfolio_value = self.initial_cash
        return self._get_observation(), {}

    def _get_observation(self):
        feats = [self._normalized_features[s].iloc[self.current_step].values.astype(np.float32)
                 for s in self.symbols]
        context = []
        for s in self.symbols:
            context.append(self.positions[s])
            context.append(self.unrealized_pnl[s] / self.initial_cash)
        context.append(self.cash / self.initial_cash)
        return np.concatenate(feats + [np.array(context, dtype=np.float32)])

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        total_step_return = 0.0
        total_cost = 0.0
        portfolio_vol = float(np.mean([self._volatility[s].iloc[self.current_step] for s in self.symbols]))

        for i, symbol in enumerate(self.symbols):
            target_position = float(action[i])
            current_price = self._close[symbol].iloc[self.current_step]
            old_position = self.positions[symbol]

            cost = abs(target_position - old_position) * self.initial_cash * self.transaction_fee
            total_cost += cost

            if self.current_step > 0:
                prev_price = self._close[symbol].iloc[self.current_step - 1]
                pct_change = (current_price - prev_price) / prev_price
                step_return = old_position * pct_change * self.initial_cash
            else:
                step_return = 0.0
            total_step_return += step_return

            if target_position != old_position:
                self.entry_prices[symbol] = current_price if target_position != 0 else None
            self.positions[symbol] = target_position

            if self.positions[symbol] != 0 and self.entry_prices[symbol] is not None:
                self.unrealized_pnl[symbol] = (
                    self.positions[symbol] * self.initial_cash
                    * (current_price - self.entry_prices[symbol]) / self.entry_prices[symbol]
                )
            else:
                self.unrealized_pnl[symbol] = 0.0

        self.cash += total_step_return - total_cost
        self.portfolio_value = self.cash

        mean_abs_position = float(np.mean([abs(p) for p in self.positions.values()]))
        reward = (total_step_return - total_cost) / self.initial_cash
        reward -= self.risk_lambda * mean_abs_position * portfolio_vol

        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1
        truncated = False
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, float(reward), done, truncated, {
            "portfolio_value": self.portfolio_value,
            "positions": dict(self.positions),
        }

    def render(self):
        print(f"Step: {self.current_step} | Positions: {self.positions} | Value: {self.portfolio_value:.2f}")
