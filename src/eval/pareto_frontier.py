"""Tier 2, Phase 3 -- Pareto frontier: reframes the CVaR "failure"
(docs/risk_sensitivity_ablation.md single-window, docs/
walk_forward_upgrade_validation.md walk-forward) as a measured tradeoff
instead of a dead end.

WHY THIS EXISTS: CVaR lost to plain PPO on walk-forward Sharpe (0.110 vs
0.712 basket average) -- but that's exactly what you'd expect if CVaR is
genuinely trading expected return for lower tail risk, not evidence it's
simply broken. A single scalarized Sharpe number can't distinguish those
two explanations. This sweeps `CVaRPPO`'s own `cvar_alpha` parameter
(src/agents/distributional_critic.py -- already-tested infrastructure,
no new RL algorithm needed) from near-1.0 (averaging almost all predicted
quantiles -> close to a plain expected-value baseline) down to 0.1
(averaging only the worst 10% of quantiles -> aggressive tail-focus),
producing a genuine risk/return dial instead of one fixed (alpha=0.2)
config, and plots walk-forward Sharpe against walk-forward max drawdown
across that sweep -- the actual tradeoff CVaR was designed to make
visible, per-asset, rather than collapsed into a single number.

python -m src.eval.pareto_frontier [--assets GME AMC] [--alphas 0.9 0.7 0.5 0.3 0.1]
"""
import argparse
from pathlib import Path

import pandas as pd

from src.agents.common import get_paths
from src.eval.walk_forward import (
    audit_windows_no_leakage,
    evaluate_cvar_roll,
    generate_rolling_windows,
)


def sweep_asset(symbol, alphas, ppo_timesteps=15_000, seed=42,
                 train_months=6, val_months=1, step_months=1, limit_rolls=None):
    """Runs evaluate_cvar_roll (unmodified, from src/eval/walk_forward.py)
    at each `cvar_alpha` in `alphas`, across ALL of `symbol`'s walk-forward
    rolls, and returns the mean Sharpe/Sortino/max-drawdown per alpha --
    the same walk-forward protocol as every other RL variant in this
    project, not a repeat of the single-window-only mistake the original
    CVaR/architecture ablations made."""
    features_path, price_path = get_paths(symbol)
    feat = pd.read_csv(features_path, index_col="date", parse_dates=True)
    price = pd.read_csv(price_path, index_col="date", parse_dates=True)
    joined_index = feat.join(price[["Close"]], how="inner").sort_index().index
    windows = generate_rolling_windows(joined_index, train_months, val_months, step_months)
    if limit_rolls is not None:
        windows = windows[:limit_rolls]
    audit_windows_no_leakage(windows)

    rows = []
    for alpha in alphas:
        print(f"  {symbol}, cvar_alpha={alpha}: {len(windows)} rolls")
        roll_metrics = []
        for train_start, train_end, val_start, val_end in windows:
            metrics = evaluate_cvar_roll(symbol, train_start, train_end, val_start, val_end,
                                          ppo_timesteps, seed, cvar_alpha=alpha)
            roll_metrics.append(metrics)
        roll_df = pd.DataFrame(roll_metrics)
        rows.append({
            "asset": symbol, "cvar_alpha": alpha,
            "Sharpe Ratio": roll_df["Sharpe Ratio"].mean(),
            "Sortino Ratio": roll_df["Sortino Ratio"].mean(),
            "Max Drawdown": roll_df["Max Drawdown"].mean(),
            "Cumulative Return": roll_df["Cumulative Return"].mean(),
            "n_rolls": len(roll_df),
        })
    return pd.DataFrame(rows)


# Reference points from already-completed walk-forward runs (NOT
# recomputed here -- copied from docs/walk_forward_upgrade_validation.md
# so the frontier can be plotted against them without retraining).
REFERENCE_POINTS = {
    "GME": {"Original PPO": {"Sharpe Ratio": -0.9995, "Max Drawdown": -0.1296},
            "CVaR (alpha=0.2, original)": {"Sharpe Ratio": -0.5626, "Max Drawdown": -0.073}},
    "AMC": {"Original PPO": {"Sharpe Ratio": 0.2780, "Max Drawdown": -0.2314},
            "CVaR (alpha=0.2, original)": {"Sharpe Ratio": -0.4490, "Max Drawdown": -0.1546}},
    "DOGE": {"Original PPO": {"Sharpe Ratio": 1.7116, "Max Drawdown": -0.0988},
             "CVaR (alpha=0.2, original)": {"Sharpe Ratio": 0.8747, "Max Drawdown": -0.0553}},
    "PEPE": {"Original PPO": {"Sharpe Ratio": 1.0922, "Max Drawdown": -0.2252},
             "CVaR (alpha=0.2, original)": {"Sharpe Ratio": -0.1507, "Max Drawdown": -0.237}},
    "SHIB": {"Original PPO": {"Sharpe Ratio": 1.4555, "Max Drawdown": -0.0879},
             "CVaR (alpha=0.2, original)": {"Sharpe Ratio": 0.6088, "Max Drawdown": -0.059}},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", default=["GME", "AMC"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.9, 0.7, 0.5, 0.3, 0.1])
    parser.add_argument("--ppo-timesteps", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-rolls", type=int, default=None)
    parser.add_argument("--output", default="docs/pareto_frontier.md")
    args = parser.parse_args()

    all_rows = []
    for symbol in args.assets:
        df = sweep_asset(symbol, args.alphas, ppo_timesteps=args.ppo_timesteps, seed=args.seed,
                          limit_rolls=args.limit_rolls)
        all_rows.append(df)
    results = pd.concat(all_rows, ignore_index=True)

    print("\n" + "=" * 60)
    print(results.round(4).to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Pareto Frontier -- CVaR's Risk/Return Tradeoff, Made Visible\n\n")
        f.write(
            f"`CVaRPPO`'s own `cvar_alpha` parameter swept across "
            f"{args.alphas}, walk-forward evaluated ({args.ppo_timesteps} timesteps/roll, "
            f"seed={args.seed}) on {args.assets}. See src/eval/pareto_frontier.py's module "
            "docstring for why this reuses existing, already-tested CVaRPPO infrastructure "
            "rather than a new algorithm.\n\n"
        )
        for symbol in args.assets:
            f.write(f"## {symbol}\n\n")
            asset_df = results[results["asset"] == symbol].sort_values("cvar_alpha", ascending=False)
            f.write(asset_df[["cvar_alpha", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
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
