"""Phase 6 -- feature attribution & cross-platform lead-lag analysis.

python -m src.eval.attribution [--asset DOGE] [--model-path models/ppo_DOGE/model.zip]
                                [--n-repeats 5] [--max-lag 14] [--output-dir docs/figures]

Two independent analyses, run and reported separately (see module-level
SCOPE NOTE below for why they cover different asset sets):

1. Permutation importance for the trained PPO policy's actions with respect
   to its input (attention) features, on a validation window. SHAP was
   considered first (it's in requirements.txt and importable in this repo's
   venv), but SHAP's kernel/deep explainers are built around a
   feed-forward-with-fixed-output-semantics model (e.g. class probabilities
   for a classifier); wiring one up cleanly for an SB3 continuous-action
   Gaussian policy (predict() returns a deterministic mean action, not a
   distribution SHAP's explainers expect to attribute) would need a custom
   wrapper with no real payoff over the simpler approach, so permutation
   importance is used instead, per the dev plan's own explicit fallback
   ("permutation importance alone is fine and should be justified as such").
   Implementation: run one deterministic rollout, freeze the resulting
   observation matrix, then for each attention-feature column, shuffle
   (permute in time) just that column, replay the policy on the shuffled
   observations to get a counterfactual action sequence, and measure (a) how
   much the actions changed and (b) what Sharpe ratio results from actually
   taking that counterfactual action sequence through the environment.
   Repeated over several random permutations per feature and averaged, the
   same logic sklearn's permutation_importance uses for supervised models.

2. Cross-platform lead-lag: cross-correlation of RAW (pre-derivative,
   pre-normalization) attention signals -- e.g. Wikipedia `pageviews` vs.
   Google Trends `search_interest` vs. Hacker News `hn_story_count` -- at a
   range of day lags, per asset, grouped by asset class (meme stock / crypto
   / NFT). Does NOT require a trained policy at all, just the processed
   feature CSVs, so it covers every asset with processed features
   (AMC/GME/DOGE/PEPE/SHIB/boredapeyachtclub/cryptopunks), unlike (1).

SCOPE NOTE (honesty about what's actually trained -- see dev plan Phase 6
prompt's "break results down per asset class" ask): only models/ppo_DOGE
exists (Phase 4a trained one PPO policy on one asset; Phase 4b/RecurrentPPO
was a deliberate scope-cut; Phase 5's walk-forward retrains PPO per roll but
never persists those checkpoints). There is therefore no trained policy for
a meme-stock or NFT representative asset to run permutation importance on.
Rather than train new throwaway policies just to have "three bars in a
chart" (the dev plan's own Phase 11 fallback list treats trimming scope
under time pressure as expected, not a failure), analysis (1) is honestly
scoped to DOGE (crypto) only, clearly labeled as such wherever it's
reported. Analysis (2) is NOT limited this way and explicitly covers all
three asset classes, since it needs no trained model -- this is the
half of the dev plan's ask that can honestly be answered in full.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO

from src.agents.common import get_paths, get_train_val_split, select_default_asset
from src.agents.train_ppo import make_env, portfolio_values_to_returns
from src.baselines.momentum import calculate_metrics


# ─── 1. Permutation importance on the trained policy ───────────────────────


def collect_rollout_observations(model, env):
    """Deterministic rollout that also returns the raw observation vectors
    fed to the policy at each step (src.agents.train_ppo.rollout_deterministic
    only returns actions/portfolio_values, not observations -- attribution
    needs the observations themselves to permute). Returns
    (observations: (T, obs_dim) array, actions: (T,) array,
    portfolio_values: (T+1,) array -- the same convention
    rollout_deterministic uses, including the initial cash at step 0).
    """
    obs, info = env.reset()
    observations = []
    actions = []
    portfolio_values = [env.portfolio_value]
    done = False
    while not done:
        observations.append(np.asarray(obs, dtype=np.float32))
        action, _ = model.predict(obs, deterministic=True)
        actions.append(float(np.asarray(action).flatten()[0]))
        obs, reward, done, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
    return np.array(observations), np.array(actions), np.array(portfolio_values)


def replay_actions(env, action_sequence):
    """Steps a fresh episode of `env` using a GIVEN action sequence (rather
    than querying a policy at each step) -- used to measure what actually
    would have happened, portfolio-wise, had the agent taken a counterfactual
    (permutation-perturbed) action sequence. Returns the portfolio_value
    series (len(action_sequence) + 1, including initial cash), truncated
    early if the env's own episode length is shorter than the given
    sequence (shouldn't happen when replaying a same-env baseline sequence,
    but guarded rather than assumed).
    """
    env.reset()
    portfolio_values = [env.portfolio_value]
    for a in action_sequence:
        obs, reward, done, truncated, info = env.step(np.array([a], dtype=np.float32))
        portfolio_values.append(info["portfolio_value"])
        if done:
            break
    return np.array(portfolio_values)


def permutation_importance_for_policy(model, env, n_repeats=5, seed=42):
    """Permutation importance of a trained SB3 policy's ATTENTION features
    (env.feature_columns -- excludes the 3 portfolio-context observation
    dims, which are state, not an input signal to attribute importance to)
    against its own action outputs and the resulting portfolio Sharpe.

    For each feature column: shuffle that column's values across time in the
    baseline observation matrix (all other columns, including portfolio
    context, held at their real recorded values -- the standard permutation-
    importance approximation of "how much does the model's output depend on
    this input", same logic as sklearn.inspection.permutation_importance),
    get the policy's action at each shuffled observation, then actually
    replay that counterfactual action sequence through a fresh episode of
    `env` to get a real resulting Sharpe ratio. Repeated `n_repeats` times
    with independent random permutations and averaged, to reduce the
    variance of any single shuffle.

    Returns (importance_df, baseline_metrics) where importance_df has one
    row per feature, columns: feature, mean_action_shift (mean absolute
    change in action vs. the real rollout's action at that step -- higher
    means the policy relies on this feature more), std_action_shift,
    mean_sharpe_drop (baseline Sharpe minus the counterfactual-replay
    Sharpe, averaged over repeats -- positive means permuting this feature
    hurt realized performance, i.e. the feature was load-bearing for the
    policy's actual edge, not just for how much its raw action number
    moved), sorted by mean_action_shift descending.
    """
    feature_names = list(env.feature_columns)
    observations, baseline_actions, baseline_portfolio = collect_rollout_observations(model, env)
    baseline_returns = portfolio_values_to_returns(baseline_portfolio)
    baseline_metrics = calculate_metrics(baseline_returns)

    rng = np.random.default_rng(seed)
    rows = []
    for j, name in enumerate(feature_names):
        action_shifts = []
        sharpe_drops = []
        for _ in range(n_repeats):
            permuted_obs = observations.copy()
            permuted_obs[:, j] = rng.permutation(permuted_obs[:, j])

            permuted_actions, _ = model.predict(permuted_obs, deterministic=True)
            permuted_actions = np.asarray(permuted_actions).flatten()
            action_shifts.append(float(np.mean(np.abs(permuted_actions - baseline_actions))))

            replayed_values = replay_actions(env, permuted_actions)
            replayed_returns = portfolio_values_to_returns(replayed_values)
            if len(replayed_returns) >= 2:
                replayed_metrics = calculate_metrics(replayed_returns)
                sharpe_drops.append(baseline_metrics["Sharpe Ratio"] - replayed_metrics["Sharpe Ratio"])

        rows.append({
            "feature": name,
            "mean_action_shift": float(np.mean(action_shifts)),
            "std_action_shift": float(np.std(action_shifts)),
            "mean_sharpe_drop": float(np.mean(sharpe_drops)) if sharpe_drops else float("nan"),
        })

    importance_df = pd.DataFrame(rows).sort_values("mean_action_shift", ascending=False).reset_index(drop=True)
    return importance_df, baseline_metrics


def plot_feature_importance(importance_df, asset, output_path, top_n=15):
    """Bar chart of mean_action_shift per feature (dev plan Phase 6's
    explicit ask: "a bar chart of average feature importance"). Saved as a
    PNG, not shown interactively -- this module is meant to run headless
    (`python -m src.eval.attribution`), same convention as
    src/eval/walk_forward.py writing a markdown report rather than opening a
    window.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = importance_df.head(top_n).iloc[::-1]  # reverse so the largest bar is on top

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["mean_action_shift"], color="#4C72B0")
    ax.set_xlabel("Mean |action shift| under feature permutation")
    ax.set_title(f"PPO policy feature importance -- {asset} (permutation importance)")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ─── 2. Cross-platform lead-lag (raw, pre-derivative signals) ──────────────


# Raw, pre-derivative attention signals present in data/processed/<symbol>_
# features.csv for every asset in this run (see docs/feature_dictionary.md's
# "Raw / passthrough columns" table). Reddit-derived raw columns
# (post_count, comment_count, sentiment) are deliberately NOT listed here --
# they require live Reddit API credentials that are a documented, currently-
# blocked gap (configs/assets.yaml), and are in fact absent from every
# data/processed/*_features.csv in this run (verified: none of the 7
# processed files contain a `comment_count` or `sentiment` column). Using
# them here would mean silently analyzing data that doesn't exist.
RAW_SIGNALS = ["pageviews", "search_interest", "hn_story_count"]


def load_asset_class_map(config_path="configs/assets.yaml"):
    """symbol -> asset_class ('meme_stocks' / 'small_cap_crypto' /
    'nft_collections'), read from configs/assets.yaml's own category keys
    (not re-derived/hardcoded, so it can't drift from the actual config).
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    mapping = {}
    for asset_class, entries in cfg.items():
        for entry in entries:
            symbol = entry.get("ticker") or entry.get("symbol") or entry.get("id")
            mapping[symbol] = asset_class
    return mapping


def cross_correlation(series_a, series_b, max_lag=14):
    """Pearson cross-correlation between two aligned, equal-length series at
    day lags -max_lag..+max_lag. At lag k, correlates series_a[t] against
    series_b[t-k] (i.e. series_b.shift(k)): a positive k with a strong
    correlation means series_b's value from k days ago aligns with today's
    series_a -- series_b LEADS series_a by k days. A negative k with a
    strong correlation means series_a leads series_b by |k| days instead.
    Returns a pandas Series indexed by lag k. A lag with fewer than 10
    overlapping valid observations, or with either series constant over
    that overlap (undefined correlation), is NaN rather than a fabricated
    0 -- callers should skip NaN lags, not treat them as "no relationship".
    """
    a = pd.Series(series_a).reset_index(drop=True)
    b = pd.Series(series_b).reset_index(drop=True)
    n = min(len(a), len(b))
    a, b = a.iloc[:n], b.iloc[:n]

    corrs = {}
    for k in range(-max_lag, max_lag + 1):
        shifted_b = b.shift(k)
        valid = a.notna() & shifted_b.notna()
        if valid.sum() < 10:
            corrs[k] = float("nan")
            continue
        a_valid, b_valid = a[valid], shifted_b[valid]
        if a_valid.std() == 0 or b_valid.std() == 0:
            corrs[k] = float("nan")
            continue
        corrs[k] = float(a_valid.corr(b_valid))
    return pd.Series(corrs, name="correlation")


def best_lag(corr_series):
    """(lag, correlation) of the largest-magnitude, non-NaN entry in a
    cross_correlation() result. Returns (None, float('nan')) if every lag
    was NaN (e.g. one of the two signals was constant over the whole
    window)."""
    valid = corr_series.dropna()
    if valid.empty:
        return None, float("nan")
    lag = valid.abs().idxmax()
    return int(lag), float(valid.loc[lag])


def lead_lag_for_asset(symbol, signal_a, signal_b, max_lag=14, processed_dir="data/processed"):
    """Runs cross_correlation() on two RAW signal columns of one asset's
    processed features file. Returns None if either column isn't present
    for this asset (e.g. an asset with no Hacker News hits at all would
    still have the hn_story_count COLUMN with all zeros, so "missing" here
    means the column itself is absent, which in practice only ever happens
    for Reddit-only columns that were never computed -- see RAW_SIGNALS'
    comment). Otherwise returns a dict with asset, signal_a, signal_b,
    best_lag, best_corr, n_obs.
    """
    path = Path(processed_dir) / f"{symbol}_features.csv"
    df = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
    if signal_a not in df.columns or signal_b not in df.columns:
        return None

    corr_series = cross_correlation(df[signal_a], df[signal_b], max_lag=max_lag)
    lag, corr = best_lag(corr_series)
    return {
        "asset": symbol,
        "signal_a": signal_a,
        "signal_b": signal_b,
        "best_lag_days": lag,
        "best_corr": corr,
        "n_obs": int(len(df)),
    }


def summarize_lead_lag_across_assets(assets=None, signal_pairs=None, max_lag=14,
                                       processed_dir="data/processed", config_path="configs/assets.yaml"):
    """Runs lead_lag_for_asset() for every (asset, signal pair) combination
    and tags each row with its asset_class from configs/assets.yaml, so the
    dev plan's specific ask ("does a lead-lag pattern differ by asset
    class") can be read directly off the resulting table (group by
    asset_class, compare best_lag_days/best_corr). Defaults to every asset
    with a processed features file, and every pair of RAW_SIGNALS -- this
    deliberately includes the NFT assets (boredapeyachtclub, cryptopunks),
    which have no price data and so were excluded from the Phase 5 RL
    backtest, but DO have real processed attention features, so they belong
    in this analysis (which needs no price/policy at all).
    """
    if assets is None:
        assets = sorted(p.stem.removesuffix("_features") for p in Path(processed_dir).glob("*_features.csv"))
    if signal_pairs is None:
        signal_pairs = [
            (RAW_SIGNALS[i], RAW_SIGNALS[j])
            for i in range(len(RAW_SIGNALS))
            for j in range(i + 1, len(RAW_SIGNALS))
        ]

    asset_class_map = load_asset_class_map(config_path)

    rows = []
    for symbol in assets:
        for signal_a, signal_b in signal_pairs:
            result = lead_lag_for_asset(symbol, signal_a, signal_b, max_lag=max_lag, processed_dir=processed_dir)
            if result is None:
                continue
            result["asset_class"] = asset_class_map.get(symbol, "unknown")
            rows.append(result)

    return pd.DataFrame(rows)


# ─── 3. Orchestration ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default=None, help="Asset for permutation importance (default: longest-history asset with a trained model, else DOGE)")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--max-lag", type=int, default=14)
    parser.add_argument("--output-dir", default="docs/figures")
    parser.add_argument("--report", default="docs/attribution_report.md")
    args = parser.parse_args()

    asset = args.asset or select_default_asset()
    model_path = Path(args.model_path or f"models/ppo_{asset}/model.zip")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = ["# Phase 6 -- Feature Attribution & Cross-Platform Lead-Lag\n"]

    # --- Part 1: permutation importance (only where a trained policy exists) ---
    if model_path.exists():
        features_path, price_path = get_paths(asset)
        _, val_start, val_end = get_train_val_split(asset, args.train_frac)
        val_env = make_env(asset, features_path, price_path, start_date=val_start, end_date=val_end)

        model = PPO.load(model_path)
        importance_df, baseline_metrics = permutation_importance_for_policy(
            model, val_env, n_repeats=args.n_repeats, seed=42
        )

        chart_path = output_dir / f"attribution_feature_importance_{asset}.png"
        plot_feature_importance(importance_df, asset, chart_path)

        print(f"Permutation importance ({asset}, validation {val_start.date()}->{val_end.date()}):")
        print(importance_df.to_string(index=False))
        print(f"\nBaseline Sharpe on this window: {baseline_metrics['Sharpe Ratio']:.4f}")
        print(f"Saved chart to {chart_path}")

        report_lines.append(
            f"## 1. Permutation importance -- {asset} only\n\n"
            f"Only `models/ppo_{asset}/model.zip` exists as a trained checkpoint "
            "(Phase 4a trained one policy on one asset; see this module's docstring "
            "for why the per-asset-class breakdown the dev plan asks for is "
            "honestly scoped to this single asset rather than fabricated for "
            "asset classes with no trained policy). This is a within-asset-class "
            "(small-cap crypto) result only, not a cross-class comparison.\n\n"
            f"Validation window: {val_start.date()} -> {val_end.date()} "
            f"({len(importance_df)} attention features). Baseline Sharpe on this "
            f"window: {baseline_metrics['Sharpe Ratio']:.4f}.\n\n"
        )
        report_lines.append(importance_df.to_markdown(index=False))
        report_lines.append(f"\n\n![feature importance]({chart_path.as_posix()})\n")
    else:
        print(f"No trained model at {model_path} -- skipping permutation importance.")
        report_lines.append(
            f"## 1. Permutation importance\n\nSkipped: no trained model found at "
            f"`{model_path}`.\n"
        )

    # --- Part 2: cross-platform lead-lag, all asset classes ---
    lead_lag_df = summarize_lead_lag_across_assets(max_lag=args.max_lag)
    print("\nCross-platform lead-lag (raw signals, all asset classes):")
    print(lead_lag_df.to_string(index=False))

    lead_lag_csv = output_dir / "leadlag_cross_correlation.csv"
    lead_lag_df.to_csv(lead_lag_csv, index=False)

    report_lines.append(
        "\n## 2. Cross-platform lead-lag (raw, pre-derivative signals)\n\n"
        "No trained policy is required for this analysis, so it covers every "
        "asset with processed features, across all three asset classes, "
        "including the two NFT collections (which have no price data and so "
        "were excluded from the Phase 5 RL backtest). Reddit-derived raw "
        "signals (`post_count`, `comment_count`, `sentiment`) are not used -- "
        "Reddit ingestion requires live API credentials that remain a "
        "documented, blocked gap (see `configs/assets.yaml`), and none of "
        "the processed feature files in this run actually contain those "
        "columns. The raw signals actually compared instead: Wikipedia "
        "`pageviews`, Google Trends `search_interest`, and Hacker News "
        f"`hn_story_count`, at lags of -{args.max_lag}..+{args.max_lag} days.\n\n"
    )
    report_lines.append(lead_lag_df.to_markdown(index=False))
    report_lines.append(f"\n\n(Full table also saved to `{lead_lag_csv}`.)\n")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
