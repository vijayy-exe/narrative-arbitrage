"""Tier 2 follow-up -- sweeps src/agents/uncertainty_critic.py's own `k`
scaling coefficient the way src/eval/pareto_frontier.py swept CVaR's
cvar_alpha. docs/uncertainty_sizing_validation.md's own "Honest scope
note" flagged this as the natural next step, not attempted there given
time budget -- only k=2.0 had been tried, on GME and AMC.

EFFICIENCY NOTE: k only enters scale_action_by_uncertainty /
rollout_uncertainty_aware at ROLLOUT time -- train_critic_ensemble's
critics and the base PPO model (src/agents/uncertainty_critic.py) are
otherwise completely independent of k. So unlike a naive sweep that would
retrain the full (n_critic_members+1)x-cost model stack for every k value
(as src/eval/walk_forward.py's evaluate_uncertainty_aware_roll does when
called once per k), this trains each roll's base model + critic ensemble
ONCE and reuses them across every k in the sweep, paying only the cheap
inference-only rollout_uncertainty_aware cost per extra k. The ROLLOUT
TRAJECTORY itself still depends on k (a differently-scaled action changes
the env's actual next state/portfolio), so each k still needs its own
rollout call -- only the expensive training is shared, not the
evaluation. Base-model/critic-ensemble seeding exactly mirrors
evaluate_uncertainty_aware_roll's (base_model seed=seed, critic ensemble
base_seed=seed too) so this is a faithful extension of the
already-validated k=2.0 method, not a different one.

python -m src.eval.uncertainty_sweep [--assets GME AMC] [--ks 0.5 1.0 2.0 3.0 4.0]
"""
import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.common import get_paths
from src.agents.train_ppo import make_env, portfolio_values_to_returns
from src.agents.uncertainty_critic import rollout_uncertainty_aware, train_critic_ensemble
from src.baselines.momentum import calculate_metrics
from src.eval.walk_forward import audit_windows_no_leakage, compute_trade_stats, generate_rolling_windows


def sweep_asset(symbol, ks, n_critic_members=3, ppo_timesteps=15_000, seed=42,
                 train_months=6, val_months=1, step_months=1, limit_rolls=None):
    """Runs the walk-forward protocol once per roll (training a base PPO
    model + an n_critic_members-critic ensemble), then evaluates
    rollout_uncertainty_aware at every k in `ks` against that SAME trained
    pair -- see module docstring for why this is cheap relative to a naive
    per-k retrain."""
    features_path, price_path = get_paths(symbol)
    feat = pd.read_csv(features_path, index_col="date", parse_dates=True)
    price = pd.read_csv(price_path, index_col="date", parse_dates=True)
    joined_index = feat.join(price[["Close"]], how="inner").sort_index().index
    windows = generate_rolling_windows(joined_index, train_months, val_months, step_months)
    if limit_rolls is not None:
        windows = windows[:limit_rolls]
    audit_windows_no_leakage(windows)

    print(f"{symbol}: {len(windows)} rolls, training base model + {n_critic_members}-critic "
          f"ensemble once per roll, evaluating at k in {ks}")

    per_k_roll_metrics = {k: [] for k in ks}
    for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
        print(f"  roll {i + 1}/{len(windows)}: train {train_start.date()}->{train_end.date()}, "
              f"val {val_start.date()}->{val_end.date()}")

        train_env = make_env(symbol, features_path, price_path, start_date=train_start, end_date=train_end)
        val_env = make_env(symbol, features_path, price_path, start_date=val_start, end_date=val_end)

        vec_env = DummyVecEnv([lambda env=train_env: Monitor(env)])
        base_model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0)
        base_model.learn(total_timesteps=ppo_timesteps)

        critic_ensemble = train_critic_ensemble(
            lambda: make_env(symbol, features_path, price_path, start_date=train_start, end_date=train_end),
            n_members=n_critic_members, base_seed=seed, timesteps=ppo_timesteps,
        )

        for k in ks:
            portfolio_values, actions, _disagreements = rollout_uncertainty_aware(
                base_model, critic_ensemble, val_env, k=k
            )
            returns = portfolio_values_to_returns(portfolio_values)
            metrics = calculate_metrics(returns)
            metrics.update(compute_trade_stats(actions, returns.to_numpy()))
            per_k_roll_metrics[k].append(metrics)

    rows = []
    for k in ks:
        roll_df = pd.DataFrame(per_k_roll_metrics[k])
        rows.append({
            "asset": symbol, "k": k,
            "Sharpe Ratio": roll_df["Sharpe Ratio"].mean(),
            "Sortino Ratio": roll_df["Sortino Ratio"].mean(),
            "Max Drawdown": roll_df["Max Drawdown"].mean(),
            "Cumulative Return": roll_df["Cumulative Return"].mean(),
            "n_rolls": len(roll_df),
        })
    return pd.DataFrame(rows)


# Reference points from already-completed walk-forward runs (NOT
# recomputed here -- copied from docs/uncertainty_sizing_validation.md and
# docs/walk_forward_upgrade_validation.md so the sweep can be read against
# them without retraining).
REFERENCE_POINTS = {
    "GME": {"Original PPO (no scaling)": {"Sharpe Ratio": -0.9995, "Max Drawdown": -0.1296},
            "Uncertainty-aware (k=2.0, original single-k run)": {"Sharpe Ratio": -0.481, "Max Drawdown": -0.070}},
    "AMC": {"Original PPO (no scaling)": {"Sharpe Ratio": 0.2780, "Max Drawdown": -0.2314},
            "Uncertainty-aware (k=2.0, original single-k run)": {"Sharpe Ratio": -0.650, "Max Drawdown": -0.133}},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", default=["GME", "AMC"])
    parser.add_argument("--ks", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0, 4.0])
    parser.add_argument("--n-critic-members", type=int, default=3)
    parser.add_argument("--ppo-timesteps", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-rolls", type=int, default=None)
    parser.add_argument("--output", default="docs/uncertainty_k_sweep.md")
    args = parser.parse_args()

    all_rows = []
    for symbol in args.assets:
        df = sweep_asset(symbol, args.ks, n_critic_members=args.n_critic_members,
                          ppo_timesteps=args.ppo_timesteps, seed=args.seed, limit_rolls=args.limit_rolls)
        all_rows.append(df)
    results = pd.concat(all_rows, ignore_index=True)

    print("\n" + "=" * 60)
    print(results.round(4).to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Uncertainty-Aware Sizing -- k Sweep\n\n")
        f.write(
            f"`scale_action_by_uncertainty`'s own `k` coefficient swept across "
            f"{args.ks}, walk-forward evaluated ({args.ppo_timesteps} timesteps/roll, "
            f"{args.n_critic_members} critic members, seed={args.seed}) on {args.assets}. "
            "The base PPO model and critic ensemble are trained ONCE per roll and reused "
            "across every k (see src/eval/uncertainty_sweep.py's module docstring) -- only "
            "the rollout, not the training, is repeated per k.\n\n"
        )
        for symbol in args.assets:
            f.write(f"## {symbol}\n\n")
            asset_df = results[results["asset"] == symbol].sort_values("k")
            f.write(asset_df[["k", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
                               "Cumulative Return"]].round(4).to_markdown(index=False))
            f.write("\n\n")
            if symbol in REFERENCE_POINTS:
                f.write("Reference points (already-completed walk-forward runs, not recomputed here):\n\n")
                for label, metrics in REFERENCE_POINTS[symbol].items():
                    f.write(f"- **{label}**: Sharpe {metrics['Sharpe Ratio']:.4f}, "
                            f"Max Drawdown {metrics['Max Drawdown']:.4f}\n")
                f.write("\n")
    print(f"\nSaved report to {output_path}")
    results.to_csv(output_path.with_suffix(".csv"), index=False)
    return results


if __name__ == "__main__":
    main()
