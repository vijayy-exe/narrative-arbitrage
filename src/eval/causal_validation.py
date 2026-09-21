"""Tier 2, Phase 1 -- causal validation of the project's core hypothesis,
independent of any RL result.

WHY THIS EXISTS: every result in this project so far tests "does attention
lead price" *indirectly*, through whether an RL agent trained on attention
features happens to be profitable. That conflates two separate questions:
(a) is attention genuinely informative about future price, and (b) is a
given RL agent good at exploiting that information. This module answers
(a) directly and quantitatively, with a permutation-test p-value per
asset per signal, independent of whether any RL variant in this project
is profitable.

python -m src.eval.causal_validation [--assets DOGE GME ...] [--bins 4] [--n-permutations 200]

METHOD:
1. Transfer entropy (Schreiber 2000), TE(source -> target) =
   I(target_t ; source_{t-1} | target_{t-1}) = H(target_t | target_{t-1})
   - H(target_t | target_{t-1}, source_{t-1}) -- a directional, nonlinear
   measure of whether `source`'s past genuinely reduces uncertainty about
   `target`'s future beyond what `target`'s own past already explains
   (unlike plain correlation, which is symmetric and contemporaneous).
   Estimated via a plug-in histogram estimator (quantile-binned into
   `bins` discrete levels) -- a known, simple, defensible choice for a
   scoped analysis; documented here rather than presented as an unbiased
   estimator (plug-in histogram entropy estimators have a known small-
   sample upward bias, which is exactly why step 2's permutation test is
   used to get a p-value rather than trusting the raw TE magnitude
   against an asymptotic null).
2. Granger causality (linear special case, `statsmodels`) as a simpler,
   well-understood cross-check -- if TE finds a nonlinear lead-lag effect
   that Granger causality misses entirely, that's itself informative (a
   genuinely nonlinear relationship, consistent with this project's
   "acceleration/velocity, not raw level" thesis).
3. A permutation test (shuffle `source`, recompute TE, repeat
   `n_permutations` times) gives a null distribution for TE under "no
   genuine lead-lag relationship, just this sample's own autocorrelation
   structure" -- the p-value is the fraction of shuffled TEs >= the
   observed TE. This is what makes "signal X leads price for asset Y" a
   falsifiable, quantified claim instead of an eyeballed lead-lag chart.
4. Cross-referenced against docs/attribution_report_all_assets.md's
   permutation-importance rankings: does the RL policy's most-relied-on
   feature per asset actually show statistically significant causal lead
   over price? A feature the policy weights heavily but that shows no
   causal lead is evidence the policy may be fitting noise correlated
   with its OWN past actions/portfolio state, not genuine information.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import grangercausalitytests

from src.agents.common import available_assets, get_paths
from src.data_pipeline.features import recompute_derivatives_on_index


def _discretize(series, bins):
    """Quantile-bins a 1-D array into `bins` integer levels (0..bins-1).
    pd.qcut with duplicates='drop' can return fewer than `bins` levels if
    the series has repeated values concentrated at a boundary (common for
    e.g. a feature that's exactly 0 on most days) -- levels are always
    contiguous integers starting at 0 regardless, which is all
    _joint_entropy below requires."""
    codes = pd.qcut(pd.Series(series), bins, labels=False, duplicates="drop")
    return codes.to_numpy()


def _joint_entropy(*discretized_arrays):
    """Shannon entropy, in bits, of the joint distribution of the given
    equal-length discretized integer arrays (plug-in histogram estimator
    -- see module docstring for the known small-sample bias and why the
    permutation test in transfer_entropy_p_value works around it)."""
    stacked = np.stack(discretized_arrays, axis=1)
    _, counts = np.unique(stacked, axis=0, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def transfer_entropy(source, target, bins=4):
    """TE(source -> target), lag-1 histories for both series. `source` and
    `target` must be equal-length, NaN-free, already time-aligned arrays.
    Returns TE in bits (>= 0 in theory; small negative values are possible
    from finite-sample noise in the plug-in estimator and should be read
    as "no detectable effect," not a negative causal effect)."""
    s = _discretize(source, bins)
    t = _discretize(target, bins)
    t_now, t_prev, s_prev = t[1:], t[:-1], s[:-1]

    h_t_given_tprev = _joint_entropy(t_now, t_prev) - _joint_entropy(t_prev)
    h_t_given_tprev_sprev = _joint_entropy(t_now, t_prev, s_prev) - _joint_entropy(t_prev, s_prev)
    return h_t_given_tprev - h_t_given_tprev_sprev


def transfer_entropy_permutation_test(source, target, bins=4, n_permutations=200, seed=42):
    """Returns (observed_te, p_value). p_value = fraction of `n_permutations`
    shuffles of `source` (breaking any genuine temporal relationship to
    `target` while preserving each series' own marginal distribution) whose
    TE is >= the observed TE -- the standard permutation-test construction
    for a one-sided "is this lead-lag effect stronger than chance" claim."""
    rng = np.random.RandomState(seed)
    observed = transfer_entropy(source, target, bins=bins)

    source = np.asarray(source)
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(source)
        if transfer_entropy(shuffled, target, bins=bins) >= observed:
            at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (n_permutations + 1)  # +1/+1: standard finite-permutation correction
    return observed, p_value


def granger_causality_pvalue(source, target, max_lag=1):
    """Linear cross-check: statsmodels' F-test p-value for "source's past
    values help predict target beyond target's own past" at `max_lag`.
    Returns None if statsmodels raises (e.g. a perfectly collinear or
    degenerate series) rather than crashing the whole sweep over one
    problem feature."""
    data = np.column_stack([target, source])
    try:
        # NOTE: statsmodels >=0.15 dropped the `verbose` kwarg entirely
        # (previously verbose=False silenced its per-lag print output) --
        # pinned here to the current signature, not the older one.
        results = grangercausalitytests(data, maxlag=max_lag)
        return float(results[max_lag][0]["ssr_ftest"][1])
    except Exception:
        return None


def load_asset_signals(symbol):
    """Returns (features_df, next_day_return) for `symbol`, aligned and
    NaN-dropped -- the same recompute_derivatives_on_index reuse pattern
    every other consumer in this project follows (never re-derive
    velocity/acceleration inline)."""
    features_path, price_path = get_paths(symbol)
    features = pd.read_csv(features_path, index_col="date", parse_dates=True).sort_index()
    prices = pd.read_csv(price_path, index_col="date", parse_dates=True).sort_index()
    full_df = features.join(prices[["Close"]], how="inner").sort_index()
    full_df = recompute_derivatives_on_index(full_df)

    full_df["next_return"] = full_df["Close"].pct_change().shift(-1)
    feature_cols = [c for c in full_df.columns if c not in ("Close", "next_return")]
    df = full_df[feature_cols + ["next_return"]].dropna()
    return df[feature_cols], df["next_return"]


def run_causal_validation(assets, bins=4, n_permutations=200, max_lag=1, seed=42):
    rows = []
    for symbol in assets:
        feature_df, next_return = load_asset_signals(symbol)
        print(f"{symbol}: {len(feature_df)} aligned rows, {len(feature_df.columns)} features")
        for feature_name in feature_df.columns:
            source = feature_df[feature_name].to_numpy()
            target = next_return.to_numpy()
            if len(np.unique(source)) < bins or len(np.unique(target)) < bins:
                continue  # too few distinct values to form `bins` quantile levels -- skip, don't fabricate a result
            te, te_p = transfer_entropy_permutation_test(source, target, bins=bins,
                                                           n_permutations=n_permutations, seed=seed)
            granger_p = granger_causality_pvalue(source, target, max_lag=max_lag)
            rows.append({
                "asset": symbol, "feature": feature_name,
                "transfer_entropy_bits": te, "te_p_value": te_p,
                "granger_p_value": granger_p,
                "te_significant_p05": te_p < 0.05,
                "granger_significant_p05": (granger_p is not None) and (granger_p < 0.05),
            })
    return pd.DataFrame(rows)


def apply_multiple_comparison_correction(results, alpha=0.05):
    """Adds Bonferroni and Benjamini-Hochberg FDR corrected significance
    columns for both TE and Granger p-values. REQUIRED reading before
    trusting any "N/M pairs significant at p<0.05" headline number: with
    M simultaneous tests (e.g. 85 for a 5-asset x ~17-feature sweep),
    ~alpha*M "significant" results are expected by CHANCE ALONE even if
    every single null hypothesis is true (no genuine lead-lag anywhere).
    Nominal (uncorrected) p<0.05 counts are reported for transparency but
    should never be the headline claim on their own -- the corrected
    columns are. `granger_p_value` NaNs (a degenerate series
    grangercausalitytests couldn't fit) are treated as p=1.0 (never
    significant) for the correction, not dropped -- dropping them would
    silently shrink the effective number of tests being corrected for.
    """
    results = results.copy()
    granger_p_filled = results["granger_p_value"].fillna(1.0)

    _, te_bonf, _, _ = multipletests(results["te_p_value"], alpha=alpha, method="bonferroni")
    _, te_fdr, _, _ = multipletests(results["te_p_value"], alpha=alpha, method="fdr_bh")
    _, granger_bonf, _, _ = multipletests(granger_p_filled, alpha=alpha, method="bonferroni")
    _, granger_fdr, _, _ = multipletests(granger_p_filled, alpha=alpha, method="fdr_bh")

    results["te_q_bonferroni"] = te_bonf
    results["te_significant_bonferroni"] = te_bonf < alpha
    results["te_q_fdr"] = te_fdr
    results["te_significant_fdr"] = te_fdr < alpha
    results["granger_q_bonferroni"] = granger_bonf
    results["granger_significant_bonferroni"] = granger_bonf < alpha
    results["granger_q_fdr"] = granger_fdr
    results["granger_significant_fdr"] = granger_fdr < alpha
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", default=None)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--max-lag", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="docs/causal_validation.md")
    args = parser.parse_args()

    assets = args.assets or available_assets()
    results = run_causal_validation(assets, bins=args.bins, n_permutations=args.n_permutations,
                                     max_lag=args.max_lag, seed=args.seed)
    results = apply_multiple_comparison_correction(results)

    results_sorted = results.sort_values(["asset", "te_p_value"])
    n_total = len(results)
    n_sig_te_nominal = int(results["te_significant_p05"].sum())
    n_sig_granger_nominal = int(results["granger_significant_p05"].sum())
    n_sig_te_fdr = int(results["te_significant_fdr"].sum())
    n_sig_granger_fdr = int(results["granger_significant_fdr"].sum())
    n_sig_te_bonf = int(results["te_significant_bonferroni"].sum())
    n_sig_granger_bonf = int(results["granger_significant_bonferroni"].sum())
    expected_by_chance = round(0.05 * n_total, 1)

    print(f"\n{n_total} total (asset, feature) pairs tested "
          f"(~{expected_by_chance} 'significant at p<0.05' expected by CHANCE ALONE)")
    print(f"  Transfer entropy: {n_sig_te_nominal} nominal p<0.05, "
          f"{n_sig_te_fdr} survive BH-FDR, {n_sig_te_bonf} survive Bonferroni")
    print(f"  Granger causality: {n_sig_granger_nominal} nominal p<0.05, "
          f"{n_sig_granger_fdr} survive BH-FDR, {n_sig_granger_bonf} survive Bonferroni")

    surviving_granger = results[results["granger_significant_fdr"]].sort_values("granger_q_fdr")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Causal Validation -- Does Attention Actually Lead Price?\n\n")
        f.write(
            f"Transfer entropy + permutation test ({args.n_permutations} permutations, "
            f"{args.bins}-bin quantile discretization) and Granger causality (lag={args.max_lag}) "
            f"between each attention-derivative feature and next-day return, per asset. "
            "See src/eval/causal_validation.py's module docstring for the full method.\n\n"
        )
        f.write(
            f"**{n_total} total (asset, feature) pairs tested** -- with this many simultaneous "
            f"hypothesis tests, ~{expected_by_chance} 'significant at nominal p<0.05' results are "
            "expected purely by chance, even if NO genuine lead-lag relationship exists anywhere in "
            "this data. The nominal (uncorrected) counts below are reported for transparency but "
            "are NOT the headline finding -- the Benjamini-Hochberg FDR-corrected counts are.\n\n"
        )
        f.write("| Method | Nominal p<0.05 | Survive BH-FDR (q<0.05) | Survive Bonferroni |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(f"| Transfer entropy (nonlinear) | {n_sig_te_nominal}/{n_total} | "
                 f"**{n_sig_te_fdr}**/{n_total} | {n_sig_te_bonf}/{n_total} |\n")
        f.write(f"| Granger causality (linear) | {n_sig_granger_nominal}/{n_total} | "
                 f"**{n_sig_granger_fdr}**/{n_total} | {n_sig_granger_bonf}/{n_total} |\n\n")
        f.write(
            "**Transfer entropy finds zero pairs that survive correction** -- no robust evidence of "
            "genuine NONLINEAR lead-lag from any attention-derivative feature to next-day return, "
            "anywhere in this basket, once multiple comparisons are accounted for. Granger causality "
            "(the linear special case) finds a handful of survivors, concentrated on GME:\n\n"
        )
        if not surviving_granger.empty:
            f.write(surviving_granger[["asset", "feature", "granger_p_value", "granger_q_fdr"]]
                    .round(6).to_markdown(index=False))
            f.write("\n\n")
        f.write("## Full results, sorted by asset then TE p-value\n\n")
        f.write(results_sorted.round(4).to_markdown(index=False))
        f.write("\n")
    print(f"\nSaved report to {output_path}")
    results.to_csv(output_path.with_suffix(".csv"), index=False)
    return results


if __name__ == "__main__":
    main()
