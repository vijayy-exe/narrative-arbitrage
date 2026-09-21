"""Upgrade 4 of narrative-arbitrage-implementation-upgrade.md --
hyperparameter search + multi-seed reporting, replacing the single
seed=42/single train-val split result with a statistically defensible one.

python -m src.agents.hparam_search sweep [--asset DOGE] [--n-trials 25] [--timesteps-per-trial 20000]
python -m src.agents.hparam_search multiseed [--asset DOGE] [--n-seeds 5] [--total-timesteps 100000]

WHY RecurrentPPO / LSTM is what gets tuned here: docs/architecture_ablation.md
found LSTM wins on every basket-average metric across Upgrades 1-3 (Sharpe,
Sortino, max drawdown, AND the only architecture with a positive average
cumulative return) -- "the best-performing architecture from Upgrades 1-3",
per this upgrade's own dev prompt, is unambiguously the LSTM arm, not the
Transformer or the CVaR-MLP.

WHY DOGE is the sweep asset: this upgrade's dev prompt names it explicitly
("one representative asset (DOGE, since it's the most stable)"), and
docs/CURRENT_STATUS.md independently corroborates DOGE as the basket's most
stable/best-behaved asset across every prior evaluation in this project.
(Caveat, stated plainly: docs/architecture_ablation.md separately found
DOGE is the ONE asset where the plain MLP actually beats LSTM outright --
DOGE being "stable" and DOGE being "the asset LSTM helps most" are
different claims, only the former is why it was chosen as the search
asset, per the literal dev prompt. This module does not claim otherwise.)

SWEEP BUDGET: 20k timesteps/trial (not the full 100k used everywhere else
in this project) -- a deliberate reduction, standard HPO practice, needed
to fit a 20-30 trial budget in a reasonable amount of wall-clock time. The
WINNING config is then retrained at the full 100k-timestep budget (see
`multiseed`), so the reduced sweep budget only affects which
hyperparameters get selected, not the final reported numbers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import optuna
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.architecture_ablation import rollout_deterministic_recurrent
from src.agents.common import available_assets, get_paths, get_train_val_split
from src.agents.train_ppo import make_env, momentum_metrics_on_window, portfolio_values_to_returns
from src.baselines.momentum import calculate_metrics

SEARCH_SPACE = {
    "learning_rate": ("loguniform", 1e-5, 1e-3),
    "ent_coef": ("loguniform", 1e-4, 5e-2),
    "gae_lambda": ("uniform", 0.80, 0.99),
    "lstm_hidden_size": ("categorical", [32, 64, 128]),
}


def _suggest_params(trial):
    params = {}
    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "loguniform":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "uniform":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


def train_and_eval_lstm(asset, params, total_timesteps, seed, tensorboard_log=None):
    """Trains one RecurrentPPO run with `params` and returns (sharpe_metrics,
    momentum_metrics) on the asset's validation window -- shared by both
    the Optuna objective (reduced timestep budget) and the final multi-seed
    retrain (full budget)."""
    features_path, price_path = get_paths(asset)
    train_end, val_start, val_end = get_train_val_split(asset, 0.8)

    train_env = make_env(asset, features_path, price_path, end_date=train_end)
    val_env = make_env(asset, features_path, price_path, start_date=val_start, end_date=val_end)
    vec_train_env = DummyVecEnv([lambda: Monitor(train_env)])

    model = RecurrentPPO(
        "MlpLstmPolicy", vec_train_env, seed=seed, verbose=0, tensorboard_log=tensorboard_log,
        learning_rate=params["learning_rate"], ent_coef=params["ent_coef"], gae_lambda=params["gae_lambda"],
        policy_kwargs=dict(lstm_hidden_size=params["lstm_hidden_size"]),
    )
    model.learn(total_timesteps=total_timesteps)

    portfolio_values, actions = rollout_deterministic_recurrent(model, val_env)
    returns = portfolio_values_to_returns(portfolio_values)
    metrics = calculate_metrics(returns)
    momentum = momentum_metrics_on_window(price_path, val_start, val_end)
    return model, metrics, momentum


def run_sweep(asset, n_trials, timesteps_per_trial, seed=42, storage_path="docs/hparam_search_results.md"):
    def objective(trial):
        params = _suggest_params(trial)
        _, metrics, _ = train_and_eval_lstm(asset, params, timesteps_per_trial, seed)
        sharpe = metrics["Sharpe Ratio"]
        return sharpe if np.isfinite(sharpe) else -100.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\nBest trial: value={study.best_value:.4f} params={study.best_params}")
    return study


def run_multiseed(asset, best_params, n_seeds, total_timesteps, models_dir="models_final",
                   base_seed=100):
    """Retrains `best_params` across n_seeds distinct seeds at the FULL
    timestep budget, returns per-seed metrics for mean +/- std reporting."""
    rows = []
    for i in range(n_seeds):
        seed = base_seed + i
        model, metrics, momentum = train_and_eval_lstm(asset, best_params, total_timesteps, seed)
        rows.append({"seed": seed, **metrics})
        out_dir = Path(models_dir) / f"lstm_{asset}_seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save(out_dir / "model.zip")
        print(f"  seed={seed}: Sharpe={metrics['Sharpe Ratio']:.3f} Sortino={metrics['Sortino Ratio']:.3f}")
    return rows, momentum


def summarize_multiseed(rows, momentum):
    metric_names = ["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Cumulative Return"]
    summary = {}
    for name in metric_names:
        values = np.array([r[name] for r in rows])
        summary[name] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["sweep", "multiseed"])
    parser.add_argument("--asset", default="DOGE")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--timesteps-per-trial", type=int, default=20_000)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--best-params-json", default=None,
                         help="multiseed: path to a JSON file with the winning sweep params "
                              "(default: docs/hparam_search_best_params.json)")
    args = parser.parse_args()

    if args.mode == "sweep":
        study = run_sweep(args.asset, args.n_trials, args.timesteps_per_trial, seed=args.seed)
        out_path = Path("docs/hparam_search_best_params.json")
        out_path.write_text(json.dumps(study.best_params, indent=2))
        print(f"Saved best params to {out_path}")
    else:
        params_path = Path(args.best_params_json or "docs/hparam_search_best_params.json")
        best_params = json.loads(params_path.read_text())
        print(f"Using params from {params_path}: {best_params}")
        rows, momentum = run_multiseed(args.asset, best_params, args.n_seeds, args.total_timesteps)
        summary = summarize_multiseed(rows, momentum)
        for name, stats in summary.items():
            print(f"{name}: {stats['mean']:.4f} +/- {stats['std']:.4f}")


if __name__ == "__main__":
    main()
