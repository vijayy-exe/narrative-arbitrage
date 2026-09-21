"""Upgrade 5 -- trains a single joint PPO policy on
src/env/multi_asset_env.py's MultiAssetTradingEnv, across the whole
5-asset basket at once, and compares it against the SUM of the existing
5 independent single-asset policies (models/ppo_<asset>/model.zip,
Upgrade 0/1's "none" baseline) replayed on the SAME common date window --
a fair comparison, since the independent policies were never trained or
evaluated on this shorter, intersected calendar.

python -m src.agents.train_portfolio_ppo [--total-timesteps 100000] [--seed 42]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.common import get_paths
from src.agents.train_ppo import rollout_deterministic
from src.baselines.momentum import calculate_metrics
from src.env.multi_asset_env import MultiAssetTradingEnv

BASKET = ["DOGE", "GME", "AMC", "PEPE", "SHIB"]


def make_multi_asset_env(symbols=BASKET, start_date=None, end_date=None):
    asset_paths = {s: get_paths(s) for s in symbols}
    return MultiAssetTradingEnv(asset_paths, start_date=start_date, end_date=end_date)


def get_common_train_val_split(symbols=BASKET, train_frac=0.8):
    full_env = make_multi_asset_env(symbols)
    dates = full_env.dates
    split = int(len(dates) * train_frac)
    return dates[split - 1], dates[split], dates[-1]


def rollout_portfolio_deterministic(model, env):
    obs, info = env.reset()
    portfolio_values = [env.portfolio_value]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
    return np.array(portfolio_values)


def portfolio_values_to_returns(values):
    return pd.Series(values).pct_change().dropna()


class ValidationCallback(BaseCallback):
    def __init__(self, val_env, eval_freq=5000, verbose=0):
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            values = rollout_portfolio_deterministic(self.model, self.val_env)
            metrics = calculate_metrics(portfolio_values_to_returns(values))
            self.logger.record("validation/sharpe", metrics["Sharpe Ratio"])
            self.logger.record("validation/cumulative_return", metrics["Cumulative Return"])
        return True


def independent_policies_portfolio_baseline(val_start, val_end, symbols=BASKET,
                                             models_dir="models", initial_cash_per_asset=10000.0):
    """Replays each asset's EXISTING independent checkpoint
    (models/ppo_<asset>/model.zip) on [val_start, val_end], sums their
    dollar P&L into one combined portfolio-value series -- the baseline
    the joint policy needs to beat. Uses the single-asset
    NarrativeTradingEnv/rollout_deterministic (unchanged), not the new
    multi-asset env, since these are 5 genuinely independent policies.

    DATE ALIGNMENT: each asset's own env has its OWN trading calendar
    (stocks: weekdays only; crypto: every day) -- summing portfolio-value
    arrays by POSITION rather than by actual date would silently misalign
    e.g. a stock's Monday with a crypto's Sunday. This builds each asset's
    P&L as a date-indexed Series first, then reindexes every asset onto
    the SAME common-date index the joint policy is evaluated on
    (get_common_train_val_split's dates), forward-filling gaps (a
    weekend/holiday a stock doesn't trade is "no new P&L that day" for
    that asset, not a missing observation) before summing -- a real join,
    not a positional truncation.
    """
    from src.agents.train_ppo import make_env

    common_dates = make_multi_asset_env(symbols, start_date=val_start, end_date=val_end).dates

    pnl_series = {}
    for symbol in symbols:
        features_path, price_path = get_paths(symbol)
        env = make_env(symbol, features_path, price_path, start_date=val_start, end_date=val_end)
        model = PPO.load(Path(models_dir) / f"ppo_{symbol}" / "model.zip")
        values = rollout_deterministic(model, env)[0]
        # values has len(env.df) entries (index 0 = reset state); env.df's
        # index gives the corresponding dates for every step INCLUDING the
        # initial one, matching rollout_deterministic's own convention of
        # prepending env.portfolio_value at reset.
        dates = env.df.index[: len(values)]
        pnl_series[symbol] = pd.Series(values - initial_cash_per_asset, index=dates)

    combined = pd.Series(0.0, index=common_dates)
    for symbol, series in pnl_series.items():
        combined = combined.add(series.reindex(common_dates).ffill().fillna(0.0), fill_value=0.0)
    return (combined + initial_cash_per_asset * len(symbols)).to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", default="models_portfolio")
    parser.add_argument("--tensorboard-log", default="runs")
    args = parser.parse_args()

    train_end, val_start, val_end = get_common_train_val_split(train_frac=args.train_frac)
    print(f"Joint portfolio training window: start -> {train_end.date()}")
    print(f"Validation window: {val_start.date()} -> {val_end.date()}")

    train_env = make_multi_asset_env(end_date=train_end)
    val_env = make_multi_asset_env(start_date=val_start, end_date=val_end)

    vec_train_env = DummyVecEnv([lambda: Monitor(train_env)])
    model = PPO("MlpPolicy", vec_train_env, seed=args.seed, verbose=1, tensorboard_log=args.tensorboard_log)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=ValidationCallback(val_env, eval_freq=args.eval_freq),
        tb_log_name="portfolio_ppo",
    )

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / "model.zip")
    print(f"Saved checkpoint to {models_dir / 'model.zip'}")

    joint_values = rollout_portfolio_deterministic(model, val_env)
    joint_metrics = calculate_metrics(portfolio_values_to_returns(joint_values))

    independent_values = independent_policies_portfolio_baseline(val_start, val_end)
    independent_metrics = calculate_metrics(portfolio_values_to_returns(independent_values))

    print("\n" + "=" * 60)
    print(f"Joint portfolio PPO vs. sum of 5 independent policies "
          f"({val_start.date()} -> {val_end.date()})")
    print("=" * 60)
    comparison = pd.DataFrame({
        "Joint portfolio policy": joint_metrics,
        "Sum of 5 independent policies": independent_metrics,
    }).T
    print(comparison)

    report_path = models_dir / "validation_report.md"
    with open(report_path, "w") as f:
        f.write("# Joint Portfolio PPO -- Validation Report\n\n")
        f.write(f"Train window: start -> {train_end.date()}\n\n")
        f.write(f"Validation window: {val_start.date()} -> {val_end.date()}\n\n")
        f.write(f"Total timesteps: {args.total_timesteps}, seed: {args.seed}\n\n")
        f.write("## Joint policy vs. sum of 5 independent policies (same window)\n\n")
        f.write(comparison.to_markdown())
        f.write("\n")
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
