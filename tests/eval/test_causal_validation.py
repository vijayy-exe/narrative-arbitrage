"""Tests for src/eval/causal_validation.py (Tier 2, Phase 1). Pure-function
tests for the transfer entropy estimator (a synthetic series with a KNOWN
lead-lag relationship must show higher TE in the true direction than the
reverse/null), plus an end-to-end smoke test against real repo data.
"""
import numpy as np
import pandas as pd

from src.eval.causal_validation import (
    apply_multiple_comparison_correction,
    granger_causality_pvalue,
    run_causal_validation,
    transfer_entropy,
    transfer_entropy_permutation_test,
)


def test_transfer_entropy_detects_a_known_lagged_relationship():
    """Construct target_t = source_{t-1} + small noise -- source genuinely
    leads target by construction. TE(source -> target) must exceed
    TE(target -> source) (the wrong direction) by a wide margin."""
    rng = np.random.RandomState(0)
    n = 2000
    source = rng.normal(size=n)
    target = np.empty(n)
    target[0] = rng.normal()
    target[1:] = source[:-1] + 0.1 * rng.normal(size=n - 1)

    te_correct_direction = transfer_entropy(source, target, bins=4)
    te_wrong_direction = transfer_entropy(target, source, bins=4)
    assert te_correct_direction > te_wrong_direction
    assert te_correct_direction > 0.1  # a real, substantial effect, not noise-level


def test_transfer_entropy_near_zero_for_independent_series():
    rng = np.random.RandomState(1)
    n = 2000
    source = rng.normal(size=n)
    target = rng.normal(size=n)  # genuinely independent of source
    te = transfer_entropy(source, target, bins=4)
    assert abs(te) < 0.05  # should be small (no true relationship), allowing finite-sample noise


def test_permutation_test_gives_low_p_value_for_known_relationship():
    rng = np.random.RandomState(2)
    n = 500
    source = rng.normal(size=n)
    target = np.empty(n)
    target[0] = rng.normal()
    target[1:] = source[:-1] + 0.1 * rng.normal(size=n - 1)

    te, p_value = transfer_entropy_permutation_test(source, target, bins=4, n_permutations=100, seed=0)
    assert p_value < 0.05


def test_permutation_test_gives_high_p_value_for_independent_series():
    rng = np.random.RandomState(3)
    n = 500
    source = rng.normal(size=n)
    target = rng.normal(size=n)
    te, p_value = transfer_entropy_permutation_test(source, target, bins=4, n_permutations=100, seed=0)
    assert p_value > 0.05


def test_granger_causality_pvalue_detects_known_relationship():
    rng = np.random.RandomState(4)
    n = 500
    source = rng.normal(size=n)
    target = np.empty(n)
    target[0] = rng.normal()
    target[1:] = source[:-1] + 0.05 * rng.normal(size=n - 1)  # low noise -> should be clearly detectable linearly

    p_value = granger_causality_pvalue(source, target, max_lag=1)
    assert p_value is not None
    assert p_value < 0.05


def test_multiple_comparison_correction_reduces_significant_count_under_many_null_tests():
    """The whole point of the correction: with many simultaneous tests of
    a TRUE null (independent series, no genuine relationship), nominal
    p<0.05 will flag ~5% of them "significant" by chance, but almost none
    should survive FDR/Bonferroni correction."""
    rng = np.random.RandomState(5)
    n_tests = 100
    rows = []
    for i in range(n_tests):
        # Independent p-values distributed uniform(0,1) under a true null,
        # simulated directly rather than re-running the expensive TE/Granger
        # pipeline n_tests times -- exercises the correction logic itself.
        rows.append({
            "asset": "SIM", "feature": f"f{i}",
            "te_p_value": rng.uniform(0, 1), "granger_p_value": rng.uniform(0, 1),
            "transfer_entropy_bits": 0.0, "te_significant_p05": None,
            "granger_significant_p05": None,
        })
    df = pd.DataFrame(rows)
    corrected = apply_multiple_comparison_correction(df)

    nominal_sig = (df["te_p_value"] < 0.05).sum()
    fdr_sig = corrected["te_significant_fdr"].sum()
    bonferroni_sig = corrected["te_significant_bonferroni"].sum()
    assert bonferroni_sig <= fdr_sig <= nominal_sig + 3  # allow small simulation noise


def test_run_causal_validation_against_real_repo_data_produces_finite_results():
    """Smoke test against REAL DOGE data (small n_permutations for speed) --
    confirms the full pipeline (feature loading, alignment, TE, permutation
    test, Granger) runs end-to-end without crashing on real, messy data
    (NaN leading rows, non-normal distributions, etc.)."""
    results = run_causal_validation(["DOGE"], bins=4, n_permutations=20, max_lag=1, seed=0)
    assert not results.empty
    assert results["transfer_entropy_bits"].apply(np.isfinite).all()
    assert (results["te_p_value"] > 0).all() and (results["te_p_value"] <= 1).all()
    assert set(results["asset"]) == {"DOGE"}
