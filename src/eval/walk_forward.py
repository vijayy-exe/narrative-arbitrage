"""Phase 5 -- walk-forward backtesting harness.

python -m src.eval.walk_forward [--assets DOGE GME] [--ppo-timesteps 15000]

For each asset, splits its date range into rolling windows (default: 6
months train, 1 month validate, roll forward 1 month, retrain from scratch
each roll -- the dev plan's suggested cadence), retrains PPO and refits the
supervised (XGBoost) baseline fresh at every roll using only that roll's
train window, evaluates momentum with no fitting needed, and aggregates
out-of-sample Sharpe/Sortino/max drawdown/total return/win rate/average
trade duration across all rolls. This is the real test of whether Phase 4a's
single-window result (Sharpe 3.78 on one DOGE validation window) holds up,
or was a lucky draw on one window -- see docs/backtest_report.md for output.

Compute note: PPO is retrained from scratch on every roll (not fine-tuned
forward from the previous roll's checkpoint) -- simpler to reason about and
guaranteed leak-free (no weights ever carry information from a roll's own
future), at the cost of more total compute than warm-starting would need.
--ppo-timesteps defaults much lower (15,000) than a single dedicated
training run (Phase 4a used 100,000) specifically because this cost is
multiplied by (assets x rolls) -- see the printed roll count before this
runs the full sweep, and tune down further with --limit-rolls for a quick
smoke test before committing to a full basket run.
"""
import argparse
import os

# MUST be set before stable_baselines3 (torch) or xgboost initialize their
# native thread pools -- this script is the one place in the project that
# imports and USES both in the same process (PPO retraining + XGBoost
# refitting, per roll). Confirmed by direct repro: torch-then-xgboost import
# alone is fine, but calling XGBClassifier.fit() after stable_baselines3 has
# been used crashes with SIGSEGV (an OpenMP dual-runtime conflict, not a bug
# in this project's code); XGBClassifier(n_jobs=1) alone does NOT fix it,
# only constraining OMP_NUM_THREADS globally does. Same fix as conftest.py,
# duplicated here because conftest.py only loads under pytest, and this
# module is run directly (`python -m src.eval.walk_forward`).
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from xgboost import XGBClassifier

from src.agents.common import available_assets, get_paths
from src.agents.train_ppo import make_env, portfolio_values_to_returns, rollout_deterministic
from src.baselines.momentum import calculate_metrics
from src.data_pipeline.features import recompute_derivatives_on_index


# ─── 1. Rolling window generation ──────────────────────────────────────────


def generate_rolling_windows(index, train_months=6, val_months=1, step_months=1):
    """Splits a DatetimeIndex into rolling (train_start, train_end,
    val_start, val_end) windows: train on `train_months`, validate on the
    next `val_months`, then roll forward by `step_months` and repeat. All
    boundaries are calendar-month offsets from the index's own start/end, so
    this adapts to however much history an asset actually has (see module
    docstring -- PEPE has ~13 months, the rest have ~17).
    """
    start = index.min()
    end = index.max()
    windows = []

    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.DateOffset(months=val_months) - pd.Timedelta(days=1)

        if val_end > end:
            break

        # Snap to actual available dates within each sub-window so a roll
        # never silently starts/ends on a weekend/holiday with no data.
        train_mask = (index >= train_start) & (index <= train_end)
        val_mask = (index >= val_start) & (index <= val_end)
        if train_mask.sum() >= 10 and val_mask.sum() >= 3:
            windows.append((
                index[train_mask].min(), index[train_mask].max(),
                index[val_mask].min(), index[val_mask].max(),
            ))

        train_start = train_start + pd.DateOffset(months=step_months)

    return windows


# ─── 2. Leakage audit ───────────────────────────────────────────────────────


def audit_windows_no_leakage(windows):
    """Explicit, re-checked-at-the-walk-forward-level look-ahead-bias audit
    (the dev plan asks for this as a standalone check, not just a comment):
    every roll's train window must end strictly before its own validation
    window starts, and validation windows must be non-decreasing across
    rolls (the harness never re-validates on dates it already rolled past).
    Raises AssertionError (loudly, not a silent log) on any violation.
    """
    prev_val_end = None
    for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
        assert train_start <= train_end, f"roll {i}: train_start > train_end"
        assert train_end < val_start, f"roll {i}: train window overlaps or comes after its own validation window"
        assert val_start <= val_end, f"roll {i}: val_start > val_end"
        if prev_val_end is not None:
            assert val_end >= prev_val_end, f"roll {i}: validation window moved backwards across rolls"
        prev_val_end = val_end
    return True


# ─── 3. Trade stats (win rate, average trade duration) ─────────────────────


def compute_trade_stats(position_sign_series, daily_returns):
    """position_sign_series: per-day {-1, 0, 1} (or continuous, sign() is
    taken). A "trade" is a maximal run of consecutive days with the same
    nonzero sign. Returns win_rate (fraction of trades with positive
    cumulative return over their run) and avg_trade_duration (mean run
    length in days). Both are NaN if there were zero trades (e.g. a
    strategy that never took a position in this window).
    """
    signs = np.sign(np.asarray(position_sign_series))
    returns = np.asarray(daily_returns)
    n = min(len(signs), len(returns))
    signs, returns = signs[:n], returns[:n]

    trades = []  # list of (length, cumulative_return)
    i = 0
    while i < n:
        if signs[i] == 0:
            i += 1
            continue
        j = i
        while j < n and signs[j] == signs[i]:
            j += 1
        trade_return = float(np.prod(1 + returns[i:j]) - 1)
        trades.append((j - i, trade_return))
        i = j

    if not trades:
        return {"win_rate": float("nan"), "avg_trade_duration": float("nan"), "n_trades": 0}

    win_rate = float(np.mean([r > 0 for _, r in trades]))
    avg_duration = float(np.mean([length for length, _ in trades]))
    return {"win_rate": win_rate, "avg_trade_duration": avg_duration, "n_trades": len(trades)}


# ─── 4. Per-roll evaluation: momentum, supervised, PPO ─────────────────────


def evaluate_momentum_roll(price_path, val_start, val_end, window=20):
    df = pd.read_csv(price_path, index_col="date", parse_dates=True).sort_index()
    df["returns"] = df["Close"].pct_change()
    df["momentum"] = df["Close"].pct_change(window)
    df["position"] = np.where(df["momentum"] > 0, 1, -1)
    df["strat_returns"] = df["position"].shift(1) * df["returns"]

    window_df = df.loc[val_start:val_end].dropna(subset=["strat_returns"])
    if window_df.empty:
        return None
    metrics = calculate_metrics(window_df["strat_returns"])
    metrics.update(compute_trade_stats(window_df["position"].shift(1).fillna(0), window_df["returns"]))
    return metrics


def evaluate_supervised_roll(symbol, train_start, train_end, val_start, val_end):
    features_path, price_path = get_paths(symbol)
    df_feat = pd.read_csv(features_path, index_col="date", parse_dates=True).sort_index()
    df_price = pd.read_csv(price_path, index_col="date", parse_dates=True).sort_index()

    df_price = df_price.copy()
    df_price["next_return"] = df_price["Close"].pct_change().shift(-1)
    df_price["target"] = (df_price["next_return"] > 0).astype(int)
    df = df_feat.join(df_price[["target", "next_return"]], how="inner")
    # BUG FIX (found 2026-09-07): see src/baselines/supervised.py's
    # identical fix and src/data_pipeline/features.py's
    # recompute_derivatives_on_index docstring -- df_feat's velocity/
    # acceleration columns were computed on the full calendar-day grid,
    # which silently zeroed out search_interest_velocity/_acceleration for
    # weekday-only-priced stock assets (GME, AMC) once joined against price.
    df = recompute_derivatives_on_index(df).dropna()

    train_df = df.loc[train_start:train_end]
    val_df = df.loc[val_start:val_end]
    if len(train_df) < 20 or val_df.empty:
        return None

    X_train, y_train = train_df[df_feat.columns], train_df["target"]
    X_val = val_df[df_feat.columns]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit on TRAIN rows of this roll only
    X_val_scaled = scaler.transform(X_val)

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_val_scaled)

    positions = np.where(preds == 1, 1, -1)
    strat_returns = positions * val_df["next_return"].to_numpy()

    metrics = calculate_metrics(pd.Series(strat_returns))
    metrics.update(compute_trade_stats(positions, val_df["next_return"].to_numpy()))
    return metrics


def evaluate_ppo_roll(symbol, train_start, train_end, val_start, val_end, timesteps, seed):
    features_path, price_path = get_paths(symbol)
    train_env = make_env(symbol, features_path, price_path, start_date=train_start, end_date=train_end)
    val_env = make_env(symbol, features_path, price_path, start_date=val_start, end_date=val_end)

    vec_env = DummyVecEnv([lambda: Monitor(train_env)])
    model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0)
    model.learn(total_timesteps=timesteps)

    portfolio_values, actions = rollout_deterministic(model, val_env)
    returns = portfolio_values_to_returns(portfolio_values)
    metrics = calculate_metrics(returns)
    metrics.update(compute_trade_stats(actions, returns.to_numpy()))
    return metrics


def evaluate_cvar_roll(symbol, train_start, train_end, val_start, val_end, timesteps, seed,
                        cvar_alpha=0.2, n_quantiles=32):
    """Upgrade 1's CVaR-PPO (src/agents/distributional_critic.py), retrained
    fresh per roll -- same protocol as evaluate_ppo_roll, only the model
    class/policy differ. Added 2026-09-20 to close the walk-forward gap
    docs/risk_sensitivity_ablation.md explicitly flagged as its top
    follow-up (every Upgrade-1 result was single-window until this)."""
    from src.agents.distributional_critic import CVaRPPO, QuantileCriticPolicy

    features_path, price_path = get_paths(symbol)
    train_env = make_env(symbol, features_path, price_path, start_date=train_start, end_date=train_end)
    val_env = make_env(symbol, features_path, price_path, start_date=val_start, end_date=val_end)

    vec_env = DummyVecEnv([lambda: Monitor(train_env)])
    model = CVaRPPO(QuantileCriticPolicy, vec_env, seed=seed, verbose=0,
                     policy_kwargs=dict(n_quantiles=n_quantiles, cvar_alpha=cvar_alpha))
    model.learn(total_timesteps=timesteps)

    portfolio_values, actions = rollout_deterministic(model, val_env)
    returns = portfolio_values_to_returns(portfolio_values)
    metrics = calculate_metrics(returns)
    metrics.update(compute_trade_stats(actions, returns.to_numpy()))
    return metrics


def evaluate_lstm_roll(symbol, train_start, train_end, val_start, val_end, timesteps, seed):
    """Upgrade 3's RecurrentPPO/LSTM (docs/architecture_ablation.md's
    basket-average winner), retrained fresh per roll. Added 2026-09-20 to
    close the walk-forward gap that doc explicitly flagged as its top
    follow-up. Uses architecture_ablation.py's own recurrent-state-threaded
    rollout (a plain rollout_deterministic would silently reset the LSTM's
    hidden state every step -- see that function's docstring)."""
    from sb3_contrib import RecurrentPPO

    from src.agents.architecture_ablation import rollout_deterministic_recurrent

    features_path, price_path = get_paths(symbol)
    train_env = make_env(symbol, features_path, price_path, start_date=train_start, end_date=train_end)
    val_env = make_env(symbol, features_path, price_path, start_date=val_start, end_date=val_end)

    vec_env = DummyVecEnv([lambda: Monitor(train_env)])
    model = RecurrentPPO("MlpLstmPolicy", vec_env, seed=seed, verbose=0)
    model.learn(total_timesteps=timesteps)

    portfolio_values, actions = rollout_deterministic_recurrent(model, val_env)
    returns = portfolio_values_to_returns(portfolio_values)
    metrics = calculate_metrics(returns)
    metrics.update(compute_trade_stats(actions, returns.to_numpy()))
    return metrics


def evaluate_transformer_roll(symbol, train_start, train_end, val_start, val_end, timesteps, seed,
                               lookback_window=None):
    """Upgrade 2's Transformer temporal-attention encoder
    (src/agents/attention_encoder.py), retrained fresh per roll. Added
    2026-09-20 to close the walk-forward gap docs/architecture_ablation.md
    flagged as its top follow-up. Uses a windowed NarrativeTradingEnv (see
    that env's lookback_window docstring) instead of make_env's default
    single-timestep observation -- the ONE evaluate_*_roll function here
    that can't reuse make_env unchanged, for that reason."""
    from src.agents.architecture_ablation import TRANSFORMER_LOOKBACK_WINDOW
    from src.agents.attention_encoder import TemporalAttentionExtractor
    from src.env.trading_env import NarrativeTradingEnv

    lookback_window = lookback_window or TRANSFORMER_LOOKBACK_WINDOW
    features_path, price_path = get_paths(symbol)
    train_env = NarrativeTradingEnv(symbol, features_path, price_path, start_date=train_start,
                                     end_date=train_end, lookback_window=lookback_window)
    val_env = NarrativeTradingEnv(symbol, features_path, price_path, start_date=val_start,
                                   end_date=val_end, lookback_window=lookback_window)
    n_raw_features = train_env.num_features

    vec_env = DummyVecEnv([lambda: Monitor(train_env)])
    model = PPO(
        "MlpPolicy", vec_env, seed=seed, verbose=0,
        policy_kwargs=dict(
            features_extractor_class=TemporalAttentionExtractor,
            features_extractor_kwargs=dict(lookback_window=lookback_window, n_raw_features=n_raw_features),
        ),
    )
    model.learn(total_timesteps=timesteps)

    portfolio_values, actions = rollout_deterministic(model, val_env)
    returns = portfolio_values_to_returns(portfolio_values)
    metrics = calculate_metrics(returns)
    metrics.update(compute_trade_stats(actions, returns.to_numpy()))
    return metrics


def evaluate_bc_pretrained_ppo_roll(symbol, train_start, train_end, val_start, val_end, timesteps, seed,
                                     bc_epochs=10):
    """Upgrade 6 (scoped: behavior-cloning pretraining, see
    src/agents/behavior_cloning.py's module docstring for why this and not
    a full Decision Transformer), retrained fresh per roll -- tests
    whether a momentum-pretrained warm start reduces the single-window-
    vs-walk-forward instability this project has repeatedly documented
    (e.g. DOGE: 3.95 single-window vs 1.71 walk-forward average) by
    comparing per-roll Sharpe VARIANCE against evaluate_ppo_roll's (the
    same architecture, cold-start), not just the mean."""
    from src.agents.behavior_cloning import pretrain_behavior_cloning

    features_path, price_path = get_paths(symbol)
    train_env = make_env(symbol, features_path, price_path, start_date=train_start, end_date=train_end)
    val_env = make_env(symbol, features_path, price_path, start_date=val_start, end_date=val_end)

    vec_env = DummyVecEnv([lambda: Monitor(train_env)])
    model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0)
    pretrain_behavior_cloning(model, train_env, epochs=bc_epochs, seed=seed)
    model.learn(total_timesteps=timesteps)

    portfolio_values, actions = rollout_deterministic(model, val_env)
    returns = portfolio_values_to_returns(portfolio_values)
    metrics = calculate_metrics(returns)
    metrics.update(compute_trade_stats(actions, returns.to_numpy()))
    return metrics


def evaluate_lstm_ensemble_roll(symbol, train_start, train_end, val_start, val_end, timesteps, seed,
                                 n_members=3):
    """A seed-ensemble of `n_members` independently-seeded LSTM/RecurrentPPO
    models (src/agents/ensemble.py), retrained fresh per roll -- the direct
    answer to docs/walk_forward_upgrade_validation.md's seed-robustness
    finding (GME/LSTM walk-forward Sharpe swung 1.38/1.92/-0.62 across 3
    seeds). Trains n_members models per roll (n_members x the compute of
    evaluate_lstm_roll) and evaluates by averaging their actions at every
    step -- see ensemble.py's module docstring for why this differs from
    averaging N separate backtests after the fact."""
    from src.agents.ensemble import rollout_deterministic_ensemble, train_lstm_ensemble_members

    features_path, price_path = get_paths(symbol)
    val_env = make_env(symbol, features_path, price_path, start_date=val_start, end_date=val_end)

    models = train_lstm_ensemble_members(
        lambda: make_env(symbol, features_path, price_path, start_date=train_start, end_date=train_end),
        n_members=n_members, base_seed=seed, timesteps=timesteps,
    )

    portfolio_values, actions = rollout_deterministic_ensemble(models, val_env)
    returns = portfolio_values_to_returns(portfolio_values)
    metrics = calculate_metrics(returns)
    metrics.update(compute_trade_stats(actions, returns.to_numpy()))
    return metrics


def evaluate_uncertainty_aware_roll(symbol, train_start, train_end, val_start, val_end, timesteps, seed,
                                     n_critic_members=3, uncertainty_k=2.0):
    """Tier 2, Phase 5 (src/agents/uncertainty_critic.py) -- a base PPO
    policy's action scaled down by an independently-trained critic
    ensemble's value disagreement, retrained fresh per roll. Trains
    n_critic_members + 1 models per roll (the base policy, plus the critic
    ensemble) -- (n_critic_members+1)x the compute of evaluate_ppo_roll."""
    from src.agents.uncertainty_critic import rollout_uncertainty_aware, train_critic_ensemble

    features_path, price_path = get_paths(symbol)
    train_env = make_env(symbol, features_path, price_path, start_date=train_start, end_date=train_end)
    val_env = make_env(symbol, features_path, price_path, start_date=val_start, end_date=val_end)

    vec_env = DummyVecEnv([lambda: Monitor(train_env)])
    base_model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0)
    base_model.learn(total_timesteps=timesteps)

    critic_ensemble = train_critic_ensemble(
        lambda: make_env(symbol, features_path, price_path, start_date=train_start, end_date=train_end),
        n_members=n_critic_members, base_seed=seed, timesteps=timesteps,
    )

    portfolio_values, actions, disagreements = rollout_uncertainty_aware(
        base_model, critic_ensemble, val_env, k=uncertainty_k
    )
    returns = portfolio_values_to_returns(portfolio_values)
    metrics = calculate_metrics(returns)
    metrics.update(compute_trade_stats(actions, returns.to_numpy()))
    metrics["mean_uncertainty_scale"] = float(np.mean(np.exp(-uncertainty_k * disagreements)))
    return metrics


RL_ROLL_EVALUATORS = {
    "ppo": evaluate_ppo_roll,
    "cvar": evaluate_cvar_roll,
    "lstm": evaluate_lstm_roll,
    "transformer": evaluate_transformer_roll,
    "lstm_ensemble": evaluate_lstm_ensemble_roll,
    "bc_pretrained": evaluate_bc_pretrained_ppo_roll,
    "uncertainty_aware": evaluate_uncertainty_aware_roll,
}


# ─── 5. Orchestration ────────────────────────────────────────────────────────


def run_walk_forward_for_asset(symbol, train_months=6, val_months=1, step_months=1,
                                ppo_timesteps=15_000, seed=42, limit_rolls=None,
                                rl_variant="ppo", include_baselines=True):
    """rl_variant selects which RL_ROLL_EVALUATORS entry supplies the "ppo"-
    labeled row (added 2026-09-20 to extend this harness beyond the
    original feed-forward-only PPO -- see RL_ROLL_EVALUATORS). The output
    row's `method` column is always literally "ppo" for backward
    compatibility with every existing consumer of docs/backtest_report.md's
    schema (backend/, tests/) regardless of which architecture/objective
    actually produced it -- callers that need to distinguish rl_variant
    runs must track it out-of-band (main() does, via the output path/label).
    include_baselines=False skips momentum/supervised (needed when re-
    running the SAME rolls for multiple rl_variants -- no reason to refit
    the identical baseline 4 extra times)."""
    features_path, price_path = get_paths(symbol)
    feat = pd.read_csv(features_path, index_col="date", parse_dates=True)
    price = pd.read_csv(price_path, index_col="date", parse_dates=True)
    joined_index = feat.join(price[["Close"]], how="inner").sort_index().index

    windows = generate_rolling_windows(joined_index, train_months, val_months, step_months)
    if limit_rolls is not None:
        windows = windows[:limit_rolls]
    audit_windows_no_leakage(windows)

    print(f"{symbol} [{rl_variant}]: {len(windows)} rolls "
          f"({train_months}mo train / {val_months}mo val / {step_months}mo step)")

    rl_evaluator = RL_ROLL_EVALUATORS[rl_variant]

    rows = []
    for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
        print(f"  roll {i+1}/{len(windows)}: train {train_start.date()}->{train_end.date()}, "
              f"val {val_start.date()}->{val_end.date()}")

        results = []
        if include_baselines:
            results.append(("momentum", evaluate_momentum_roll(price_path, val_start, val_end)))
            results.append(("supervised", evaluate_supervised_roll(symbol, train_start, train_end, val_start, val_end)))
        results.append(("ppo", rl_evaluator(symbol, train_start, train_end, val_start, val_end, ppo_timesteps, seed)))

        for method, metrics in results:
            if metrics is None:
                continue
            row = {"asset": symbol, "roll": i, "method": method,
                   "val_start": val_start.date().isoformat(), "val_end": val_end.date().isoformat()}
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_results(all_results):
    """Mean out-of-sample metrics per (asset, method) across rolls, plus a
    basket-wide average across assets."""
    metric_cols = ["Cumulative Return", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "win_rate", "avg_trade_duration"]
    per_asset = all_results.groupby(["asset", "method"])[metric_cols].mean().round(4)
    basket_avg = all_results.groupby("method")[metric_cols].mean().round(4)
    return per_asset, basket_avg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", default=None, help="Default: all assets with price+feature data")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--val-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--ppo-timesteps", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-rolls", type=int, default=None, help="Cap rolls per asset (for a quick smoke test)")
    parser.add_argument("--rl-variant", choices=sorted(RL_ROLL_EVALUATORS.keys()), default="ppo",
                         help="Which RL_ROLL_EVALUATORS entry supplies the 'ppo'-labeled row "
                              "(added 2026-09-20 to walk-forward-validate Upgrades 1-3).")
    parser.add_argument("--no-baselines", action="store_true",
                         help="Skip momentum/supervised (for re-running the same rolls across "
                              "multiple --rl-variant values without refitting baselines each time).")
    parser.add_argument("--output", default="docs/backtest_report.md")
    args = parser.parse_args()

    assets = args.assets or available_assets()

    all_results = []
    for symbol in assets:
        df = run_walk_forward_for_asset(
            symbol, args.train_months, args.val_months, args.step_months,
            args.ppo_timesteps, args.seed, args.limit_rolls,
            rl_variant=args.rl_variant, include_baselines=not args.no_baselines,
        )
        all_results.append(df)
    all_results = pd.concat(all_results, ignore_index=True)

    per_asset, basket_avg = aggregate_results(all_results)

    print("\n" + "=" * 60)
    print("WALK-FORWARD RESULTS -- per asset, mean across rolls")
    print("=" * 60)
    print(per_asset)
    print("\n" + "=" * 60)
    print("WALK-FORWARD RESULTS -- basket average across all assets/rolls")
    print("=" * 60)
    print(basket_avg)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# Walk-Forward Backtest Report (Phase 5)\n\n")
        f.write(
            f"Config: train={args.train_months}mo, val={args.val_months}mo, "
            f"step={args.step_months}mo, ppo_timesteps={args.ppo_timesteps}, seed={args.seed}\n\n"
        )
        f.write(
            "PPO and the supervised (XGBoost) baseline are retrained from "
            "scratch at every roll using only that roll's train window "
            "(see src/eval/walk_forward.py module docstring). Momentum "
            "needs no fitting. Every roll's train/validation boundaries "
            "are explicitly leak-audited (audit_windows_no_leakage) before "
            "any evaluation runs.\n\n"
        )
        f.write("## Per-asset results (mean across rolls)\n\n")
        f.write(per_asset.to_markdown())
        f.write("\n\n## Basket-wide average (across all assets and rolls)\n\n")
        f.write(basket_avg.to_markdown())
        f.write("\n\n## Per-roll detail\n\n")
        f.write(all_results.to_markdown(index=False))
        f.write("\n")
    print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
