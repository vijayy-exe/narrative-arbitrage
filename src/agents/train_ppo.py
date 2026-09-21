"""Phase 4a -- single-asset PPO training on NarrativeTradingEnv.

python -m src.agents.train_ppo [--asset DOGE] [--total-timesteps 50000]
                                [--risk-sensitivity {none,cvar}] [--cvar-alpha 0.2]
                                [--n-quantiles 32]

Trains PPO on a single asset's TRAIN window, logs training curves (episode
reward via SB3's Monitor wrapper, plus validation Sharpe via a custom
callback) to TensorBoard, saves a checkpoint to models/ppo_<asset>/, then
evaluates the trained policy on the held-out VALIDATION window and compares
its Sharpe/Sortino/max drawdown against the momentum baseline computed on
that *same* exact window (src/baselines/momentum.py's own report uses a
different, unrelated 80/20 split done independently inside that script, so
it is not a same-window comparison -- this script recomputes momentum on
the RL agent's own validation dates for an apples-to-apples number, and
separately reports what Phase 2's supervised.py found on its own window,
clearly labeled as a different window rather than conflating the two).

--risk-sensitivity (Upgrade 1, narrative-arbitrage-implementation-
upgrade.md): "none" (default) is byte-for-byte the ORIGINAL Phase 4a
behavior (stock PPO + "MlpPolicy") -- existing DOGE/AMC/GME/PEPE/SHIB
checkpoints and validation_report.md numbers remain reproducible from this
same script. "cvar" swaps in CVaRPPO + QuantileCriticPolicy
(src/agents/distributional_critic.py) -- a distributional critic whose
CVaR_alpha (not mean) return estimate is what SB3's GAE/rollout code uses
as the value baseline, making the policy explicitly tail-risk-averse
instead of expected-return-only. See that module's docstring for the full
mechanism and why it was added (AMC/GME's catastrophic single-window
losses, see docs/reward_hacking_findings.md).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.common import get_paths, get_train_val_split, select_default_asset
from src.agents.distributional_critic import CVaRPPO, QuantileCriticPolicy
from src.baselines.momentum import calculate_metrics
from src.env.trading_env import NarrativeTradingEnv


def build_model(risk_sensitivity, vec_train_env, seed, tensorboard_log, n_quantiles=32, cvar_alpha=0.2):
    """Selects PPO (risk_sensitivity="none", unchanged original behavior)
    vs CVaRPPO (risk_sensitivity="cvar", Upgrade 1) -- the ONLY branch point
    between the two training paths; everything else in main() is shared."""
    if risk_sensitivity == "none":
        return PPO("MlpPolicy", vec_train_env, seed=seed, verbose=1, tensorboard_log=tensorboard_log)
    elif risk_sensitivity == "cvar":
        return CVaRPPO(
            QuantileCriticPolicy, vec_train_env, seed=seed, verbose=1, tensorboard_log=tensorboard_log,
            policy_kwargs=dict(n_quantiles=n_quantiles, cvar_alpha=cvar_alpha),
        )
    raise ValueError(f"risk_sensitivity must be 'none' or 'cvar', got {risk_sensitivity!r}")


def make_env(asset, features_path, price_path, start_date=None, end_date=None):
    return NarrativeTradingEnv(
        asset_symbol=asset,
        features_path=features_path,
        price_path=price_path,
        start_date=start_date,
        end_date=end_date,
    )


def rollout_deterministic(model, env):
    """Runs `model` deterministically through one full episode of `env`
    (a raw, un-vectorized NarrativeTradingEnv). Returns the per-step
    portfolio_value series (including the initial cash at step 0) and the
    per-step action sequence -- the latter is what
    src/eval/reward_hacking_check.py inspects for policy collapse.
    """
    obs, info = env.reset()
    portfolio_values = [env.portfolio_value]
    actions = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(float(np.asarray(action).flatten()[0]))
        obs, reward, done, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
    return np.array(portfolio_values), np.array(actions)


def portfolio_values_to_returns(portfolio_values):
    return pd.Series(portfolio_values).pct_change().dropna()


class ValidationSharpeCallback(BaseCallback):
    """Every `eval_freq` training steps, runs the current policy
    deterministically on the validation env once and logs its Sharpe ratio
    to TensorBoard -- the dev plan explicitly asks for "Sharpe of the
    resulting policy on a held-out validation window" as a *logged training
    curve*, not just a one-time number printed at the end.
    """

    def __init__(self, val_env, eval_freq=5000, verbose=0):
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            portfolio_values, actions = rollout_deterministic(self.model, self.val_env)
            returns = portfolio_values_to_returns(portfolio_values)
            metrics = calculate_metrics(returns)
            self.logger.record("validation/sharpe", metrics["Sharpe Ratio"])
            self.logger.record("validation/sortino", metrics["Sortino Ratio"])
            self.logger.record("validation/cumulative_return", metrics["Cumulative Return"])
            self.logger.record("validation/mean_abs_action", float(np.mean(np.abs(actions))))
        return True


def momentum_metrics_on_window(price_path, start_date, end_date, window=20):
    """Recomputes the momentum baseline restricted to exactly
    [start_date, end_date] -- NOT src.baselines.momentum.run_momentum_strategy,
    which computes its own independent 80/20 split over the *whole* price
    history. Needed for a fair same-window comparison against the RL agent's
    validation performance.
    """
    df = pd.read_csv(price_path, index_col="date", parse_dates=True).sort_index()
    df["returns"] = df["Close"].pct_change()
    df["momentum"] = df["Close"].pct_change(window)
    df["position"] = np.where(df["momentum"] > 0, 1, -1)
    df["strat_returns"] = df["position"].shift(1) * df["returns"]
    window_returns = df.loc[start_date:end_date, "strat_returns"].dropna()
    if window_returns.empty:
        return None
    return calculate_metrics(window_returns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default=None, help="Asset symbol (default: longest-history asset)")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--tensorboard-log", default="runs")
    parser.add_argument("--risk-sensitivity", choices=["none", "cvar"], default="none",
                         help="'none' = original expected-return PPO (default, unchanged). "
                              "'cvar' = Upgrade 1's distributional/CVaR-aware PPO.")
    parser.add_argument("--cvar-alpha", type=float, default=0.2, help="Only used when --risk-sensitivity cvar.")
    parser.add_argument("--n-quantiles", type=int, default=32, help="Only used when --risk-sensitivity cvar.")
    args = parser.parse_args()

    asset = args.asset or select_default_asset()
    features_path, price_path = get_paths(asset)
    train_end, val_start, val_end = get_train_val_split(asset, args.train_frac)

    print(f"Asset: {asset}")
    print(f"Risk sensitivity: {args.risk_sensitivity}"
          + (f" (cvar_alpha={args.cvar_alpha}, n_quantiles={args.n_quantiles})" if args.risk_sensitivity == "cvar" else ""))
    print(f"Train window: start -> {train_end.date()}")
    print(f"Validation window: {val_start.date()} -> {val_end.date()}")

    train_env = make_env(asset, features_path, price_path, end_date=train_end)
    val_env = make_env(asset, features_path, price_path, start_date=val_start, end_date=val_end)

    vec_train_env = DummyVecEnv([lambda: Monitor(train_env)])

    model = build_model(
        args.risk_sensitivity, vec_train_env, args.seed, args.tensorboard_log,
        n_quantiles=args.n_quantiles, cvar_alpha=args.cvar_alpha,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=ValidationSharpeCallback(val_env, eval_freq=args.eval_freq),
        tb_log_name=f"ppo_{asset}",
    )

    models_dir = Path(args.models_dir) / f"ppo_{asset}"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.zip"
    model.save(model_path)
    print(f"Saved checkpoint to {model_path}")

    # Final evaluation on the validation window.
    portfolio_values, actions = rollout_deterministic(model, val_env)
    val_returns = portfolio_values_to_returns(portfolio_values)
    rl_metrics = calculate_metrics(val_returns)

    momentum_baseline = momentum_metrics_on_window(price_path, val_start, val_end)

    print("\n" + "=" * 60)
    print(f"PPO vs momentum baseline on {asset} validation window "
          f"({val_start.date()} -> {val_end.date()})")
    print("=" * 60)
    comparison = pd.DataFrame({
        "PPO (this run)": rl_metrics,
        "Momentum (same window)": momentum_baseline or {},
    }).T
    print(comparison)
    print(
        "\nNote: docs/baseline_reports/supervised_report.md reports "
        f"{asset}'s supervised-baseline metrics on ITS OWN 80/20 split, "
        "which is a different exact date window than this run's validation "
        "window above -- not a same-window comparison, cited for context only."
    )

    report_path = models_dir / "validation_report.md"
    with open(report_path, "w") as f:
        f.write(f"# PPO Training Report -- {asset}\n\n")
        f.write(f"Train window: start -> {train_end.date()}\n\n")
        f.write(f"Validation window: {val_start.date()} -> {val_end.date()}\n\n")
        f.write(f"Total timesteps: {args.total_timesteps}, seed: {args.seed}\n\n")
        f.write("## PPO vs momentum baseline (same validation window)\n\n")
        f.write(comparison.to_markdown())
        f.write("\n\n")
        f.write(
            f"Mean |action|: {np.mean(np.abs(actions)):.4f}, "
            f"std(action): {np.std(actions):.4f} -- see "
            "src/eval/reward_hacking_check.py for a full action-collapse check.\n"
        )
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
