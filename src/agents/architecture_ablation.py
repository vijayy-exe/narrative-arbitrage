"""Upgrades 2+3 of narrative-arbitrage-implementation-upgrade.md, run
together as the doc's own "Suggested Stopping Points" section groups them:
a controlled 3-way architecture comparison -- (a) the original flat MLP,
(b) RecurrentPPO (LSTM), (c) a Transformer temporal-attention encoder
(src/agents/attention_encoder.py) -- on identical data/reward/seed per
asset, answering "which representation of the state actually helps."

python -m src.agents.architecture_ablation --asset AMC
python -m src.agents.architecture_ablation --asset AMC --arch lstm
python -m src.agents.architecture_ablation --asset AMC --arch transformer

DESIGN DECISION -- why the LSTM arm does NOT reuse Upgrade 2's lookback
window (the implementation upgrade doc's Upgrade 3 prompt says "same
lookback-window observation from Upgrade 2"; this deliberately deviates,
documented here rather than silently): an LSTM's entire mechanism IS a
persistent hidden state carried across the whole episode -- stacking a
fixed lookback window in front of it would mean re-processing overlapping
history through TWO different memory mechanisms at once (the window's
redundant re-presentation of recent steps, AND the LSTM's own recurrence),
which conflates what's being tested rather than isolating it. This runner
gives the LSTM arm the natural lookback_window=1 (single-timestep)
observation and lets RecurrentPPO's own state carry memory across the full
episode -- a cleaner test of "does sequential memory help" than a
windowed-input LSTM would be. The MLP baseline also uses lookback_window=1
(unchanged from Upgrades 0-1). Only the Transformer arm uses a
lookback_window > 1, since a stateless network has no other way to see
temporal context.

The MLP arm is NEVER retrained here -- it reuses models/ppo_<asset>/model.zip
(Upgrade 0/1's "none" baseline) directly, evaluated on the exact same
validation window as the other two arms for a fair three-way comparison.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.attention_encoder import TemporalAttentionExtractor
from src.agents.common import available_assets, get_paths, get_train_val_split
from src.agents.train_ppo import make_env, momentum_metrics_on_window, portfolio_values_to_returns, rollout_deterministic
from src.baselines.momentum import calculate_metrics
from src.env.trading_env import NarrativeTradingEnv

TRANSFORMER_LOOKBACK_WINDOW = 10


class RecurrentValidationSharpeCallback(BaseCallback):
    """Same purpose as train_ppo.ValidationSharpeCallback, but threads LSTM
    hidden state through the validation rollout (see rollout_deterministic_
    recurrent's docstring for why this can't reuse the plain callback)."""

    def __init__(self, val_env, eval_freq=5000, verbose=0):
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            portfolio_values, actions = rollout_deterministic_recurrent(self.model, self.val_env)
            returns = portfolio_values_to_returns(portfolio_values)
            metrics = calculate_metrics(returns)
            self.logger.record("validation/sharpe", metrics["Sharpe Ratio"])
            self.logger.record("validation/sortino", metrics["Sortino Ratio"])
        return True


def rollout_deterministic_recurrent(model, env):
    """Like train_ppo.rollout_deterministic, but threads the LSTM's hidden
    state across steps via RecurrentPPO.predict()'s (state, episode_start)
    args. A plain `model.predict(obs, deterministic=True)` call (ignoring
    the returned state, the way the non-recurrent rollout_deterministic
    does) would silently reset the LSTM's hidden state on EVERY step for a
    RecurrentPPO model -- defeating the entire point of testing recurrence.
    """
    obs, info = env.reset()
    portfolio_values = [env.portfolio_value]
    actions = []
    state = None
    episode_start = np.array([True])
    done = False
    while not done:
        action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
        actions.append(float(np.asarray(action).flatten()[0]))
        obs, reward, done, truncated, info = env.step(action)
        episode_start = np.array([False])
        portfolio_values.append(info["portfolio_value"])
    return np.array(portfolio_values), np.array(actions)


def evaluate_mlp_baseline(asset, val_env):
    model_path = Path(f"models/ppo_{asset}/model.zip")
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found -- Upgrade 0/1's baseline checkpoint is required.")
    model = PPO.load(model_path)
    portfolio_values, actions = rollout_deterministic(model, val_env)
    return portfolio_values, actions


def train_lstm(asset, train_env, val_env, total_timesteps, seed, tensorboard_log, eval_freq):
    vec_train_env = DummyVecEnv([lambda: Monitor(train_env)])
    model = RecurrentPPO("MlpLstmPolicy", vec_train_env, seed=seed, verbose=1, tensorboard_log=tensorboard_log)
    model.learn(
        total_timesteps=total_timesteps,
        callback=RecurrentValidationSharpeCallback(val_env, eval_freq=eval_freq),
        tb_log_name=f"lstm_{asset}",
    )
    return model


def train_transformer(asset, train_env, val_env, total_timesteps, seed, tensorboard_log, eval_freq,
                       lookback_window=TRANSFORMER_LOOKBACK_WINDOW):
    n_raw_features = train_env.num_features
    vec_train_env = DummyVecEnv([lambda: Monitor(train_env)])
    model = PPO(
        "MlpPolicy", vec_train_env, seed=seed, verbose=1, tensorboard_log=tensorboard_log,
        policy_kwargs=dict(
            features_extractor_class=TemporalAttentionExtractor,
            features_extractor_kwargs=dict(lookback_window=lookback_window, n_raw_features=n_raw_features),
        ),
    )
    from src.agents.train_ppo import ValidationSharpeCallback
    model.learn(
        total_timesteps=total_timesteps,
        callback=ValidationSharpeCallback(val_env, eval_freq=eval_freq),
        tb_log_name=f"transformer_{asset}",
    )
    return model


def run_one_asset(asset, arch, total_timesteps=100_000, seed=42, eval_freq=5_000,
                   models_dir="models_arch", tensorboard_log="runs"):
    features_path, price_path = get_paths(asset)
    train_end, val_start, val_end = get_train_val_split(asset, 0.8)

    if arch == "mlp":
        val_env = make_env(asset, features_path, price_path, start_date=val_start, end_date=val_end)
        portfolio_values, actions = evaluate_mlp_baseline(asset, val_env)
    elif arch == "lstm":
        train_env = make_env(asset, features_path, price_path, end_date=train_end)
        val_env = make_env(asset, features_path, price_path, start_date=val_start, end_date=val_end)
        model = train_lstm(asset, train_env, val_env, total_timesteps, seed, tensorboard_log, eval_freq)
        out_dir = Path(models_dir) / f"lstm_{asset}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save(out_dir / "model.zip")
        portfolio_values, actions = rollout_deterministic_recurrent(model, val_env)
    elif arch == "transformer":
        train_env = NarrativeTradingEnv(asset, features_path, price_path, end_date=train_end,
                                         lookback_window=TRANSFORMER_LOOKBACK_WINDOW)
        val_env = NarrativeTradingEnv(asset, features_path, price_path, start_date=val_start, end_date=val_end,
                                       lookback_window=TRANSFORMER_LOOKBACK_WINDOW)
        model = train_transformer(asset, train_env, val_env, total_timesteps, seed, tensorboard_log, eval_freq)
        out_dir = Path(models_dir) / f"transformer_{asset}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save(out_dir / "model.zip")
        portfolio_values, actions = rollout_deterministic(model, val_env)
    else:
        raise ValueError(f"Unknown arch {arch!r}")

    returns = portfolio_values_to_returns(portfolio_values)
    metrics = calculate_metrics(returns)
    momentum = momentum_metrics_on_window(price_path, val_start, val_end)
    return {
        "asset": asset, "arch": arch, "metrics": metrics, "momentum": momentum,
        "mean_action": float(np.mean(actions)), "std_action": float(np.std(actions)),
        "val_start": val_start, "val_end": val_end,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default=None, help="Default: all available assets")
    parser.add_argument("--arch", choices=["mlp", "lstm", "transformer"], default=None,
                         help="Default: run all 3 architectures")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--models-dir", default="models_arch")
    parser.add_argument("--tensorboard-log", default="runs")
    args = parser.parse_args()

    assets = [args.asset] if args.asset else available_assets()
    archs = [args.arch] if args.arch else ["mlp", "lstm", "transformer"]

    rows = []
    for asset in assets:
        for arch in archs:
            print(f"\n{'=' * 60}\n{asset} / {arch}\n{'=' * 60}")
            result = run_one_asset(
                asset, arch, total_timesteps=args.total_timesteps, seed=args.seed,
                eval_freq=args.eval_freq, models_dir=args.models_dir, tensorboard_log=args.tensorboard_log,
            )
            m = result["metrics"]
            print(f"Sharpe={m['Sharpe Ratio']:.3f} Sortino={m['Sortino Ratio']:.3f} "
                  f"MaxDD={m['Max Drawdown']:.3f} CumRet={m['Cumulative Return']:.3f} "
                  f"mean_action={result['mean_action']:.3f} std_action={result['std_action']:.3f}")
            rows.append(result)

    return rows


if __name__ == "__main__":
    main()
