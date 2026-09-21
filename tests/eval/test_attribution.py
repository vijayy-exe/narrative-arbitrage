"""Tests for Phase 6's feature-attribution module (src/eval/attribution.py).

Two independent things are sanity-checked, mirroring the module's own split:
1. Permutation importance on a trained PPO policy -- uses a tiny synthetic
   asset + a tiny PPO trained for a handful of steps (same pattern as
   tests/eval/test_eval.py::test_walk_forward_runs_end_to_end_on_tiny_asset),
   so this stays fast and doesn't depend on models/ppo_DOGE existing on disk.
2. Cross-platform lead-lag cross-correlation -- checked against a synthetic
   known-lag pair (to confirm the sign/lag convention documented in
   cross_correlation()'s docstring is actually implemented correctly), and
   against this repo's REAL data/processed/*_features.csv files (to confirm
   the full multi-asset summary runs end-to-end on real data and produces
   finite, well-shaped output, not just on toy data).
"""
import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.train_ppo import make_env
from src.eval.attribution import (
    best_lag,
    collect_rollout_observations,
    cross_correlation,
    lead_lag_for_asset,
    load_asset_class_map,
    permutation_importance_for_policy,
    plot_feature_importance,
    replay_actions,
    summarize_lead_lag_across_assets,
)


# ─── cross_correlation / best_lag ────────────────────────────────────────────


def test_cross_correlation_recovers_known_lag_and_sign():
    # b is a NOISY, LAGGED copy of a: b[t] = a[t-4] + small noise, i.e. a
    # leads b by 4 days (a's past value shows up in b later). Per
    # cross_correlation()'s documented convention, that means the strongest
    # correlation should appear at a NEGATIVE lag (a leads b), around k=-4.
    rng = np.random.RandomState(0)
    n = 300
    a = pd.Series(rng.normal(size=n)).rolling(3).mean().fillna(0)
    b = a.shift(4).fillna(0) + pd.Series(rng.normal(scale=0.01, size=n))

    corr_series = cross_correlation(a, b, max_lag=10)
    assert len(corr_series) == 21  # -10..+10 inclusive
    assert corr_series.notna().all()

    lag, corr = best_lag(corr_series)
    assert lag == -4
    assert corr == pytest.approx(1.0, abs=0.05)


def test_cross_correlation_handles_constant_series_as_nan_not_zero():
    a = pd.Series([5.0] * 50)  # zero variance -- correlation is undefined
    b = pd.Series(np.arange(50, dtype=float))
    corr_series = cross_correlation(a, b, max_lag=3)
    assert corr_series.isna().all()
    lag, corr = best_lag(corr_series)
    assert lag is None
    assert np.isnan(corr)


# ─── load_asset_class_map / lead_lag_for_asset / summarize (REAL data) ─────


def test_load_asset_class_map_matches_known_real_assets():
    mapping = load_asset_class_map()
    assert mapping["DOGE"] == "small_cap_crypto"
    assert mapping["GME"] == "meme_stocks"
    assert mapping["cryptopunks"] == "nft_collections"


def test_lead_lag_for_asset_on_real_doge_data():
    result = lead_lag_for_asset("DOGE", "pageviews", "search_interest", max_lag=14)
    assert result is not None
    assert result["asset"] == "DOGE"
    assert result["n_obs"] > 100
    assert result["best_lag_days"] is not None
    assert np.isfinite(result["best_corr"])
    assert -1.0 <= result["best_corr"] <= 1.0


def test_lead_lag_for_asset_missing_column_returns_none():
    # Reddit-derived raw columns are not present in any processed file in
    # this run (documented, blocked gap) -- confirm this is handled as a
    # clean None, not an exception.
    result = lead_lag_for_asset("DOGE", "pageviews", "comment_count", max_lag=5)
    assert result is None


def test_summarize_lead_lag_across_assets_covers_all_three_classes_on_real_data():
    df = summarize_lead_lag_across_assets(max_lag=14)
    assert not df.empty
    expected_cols = {"asset", "signal_a", "signal_b", "best_lag_days", "best_corr", "n_obs", "asset_class"}
    assert expected_cols <= set(df.columns)

    # Real processed data exists for all three asset classes, including the
    # two NFT collections (excluded from the RL backtest for lack of price
    # data, but not from this feature-only analysis).
    assert set(df["asset_class"]) == {"meme_stocks", "small_cap_crypto", "nft_collections"}
    assert df["best_corr"].apply(np.isfinite).all()
    assert (df["best_corr"].abs() <= 1.0 + 1e-9).all()


# ─── permutation importance (tiny synthetic PPO) ────────────────────────────


@pytest.fixture
def tiny_asset(tmp_path, monkeypatch):
    """Same synthetic-asset pattern as test_eval.py's walk-forward smoke
    test: random-walk price + random feature columns, small enough to train
    a real (if not remotely converged) PPO policy in well under a second.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "price").mkdir(parents=True)

    n_days = 150
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rng = np.random.RandomState(0)

    close_prices = 100 + np.cumsum(rng.normal(0, 1, n_days))
    df_price = pd.DataFrame({"Close": close_prices}, index=dates)
    df_price.index.name = "date"
    df_price.to_csv(tmp_path / "data/raw/price/TESTSYM.csv")

    df_features = pd.DataFrame(
        rng.normal(size=(n_days, 4)), index=dates, columns=["sig_a", "sig_b", "sig_c", "sig_d"]
    )
    df_features.index.name = "date"
    df_features.to_csv(tmp_path / "data/processed/TESTSYM_features.csv")

    return "TESTSYM", tmp_path / "data/processed/TESTSYM_features.csv", tmp_path / "data/raw/price/TESTSYM.csv"


def _train_tiny_ppo(symbol, features_path, price_path, timesteps=128, seed=0):
    train_env = make_env(symbol, features_path, price_path)
    vec_env = DummyVecEnv([lambda: Monitor(train_env)])
    model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model


def test_collect_rollout_observations_shapes_match(tiny_asset):
    symbol, features_path, price_path = tiny_asset
    model = _train_tiny_ppo(symbol, features_path, price_path)
    val_env = make_env(symbol, features_path, price_path)

    observations, actions, portfolio_values = collect_rollout_observations(model, val_env)
    assert observations.ndim == 2
    assert observations.shape[0] == len(actions)
    assert observations.shape[1] == val_env.observation_space.shape[0]
    assert len(portfolio_values) == len(actions) + 1
    assert np.isfinite(observations).all()
    assert np.isfinite(actions).all()


def test_replay_actions_matches_baseline_when_replaying_its_own_actions(tiny_asset):
    symbol, features_path, price_path = tiny_asset
    model = _train_tiny_ppo(symbol, features_path, price_path)
    val_env = make_env(symbol, features_path, price_path)

    _, baseline_actions, baseline_portfolio = collect_rollout_observations(model, val_env)
    replayed_portfolio = replay_actions(val_env, baseline_actions)

    # Replaying the SAME action sequence through the SAME (deterministic,
    # freshly reset) env should reproduce the identical portfolio path --
    # if this didn't hold, replay_actions() would not be a valid way to
    # measure "what would have happened" under a counterfactual sequence.
    np.testing.assert_allclose(replayed_portfolio, baseline_portfolio, rtol=1e-6)


def test_permutation_importance_for_policy_well_shaped_and_no_nan(tiny_asset):
    symbol, features_path, price_path = tiny_asset
    model = _train_tiny_ppo(symbol, features_path, price_path)
    val_env = make_env(symbol, features_path, price_path)

    importance_df, baseline_metrics = permutation_importance_for_policy(model, val_env, n_repeats=2, seed=1)

    assert len(importance_df) == len(val_env.feature_columns)
    assert set(importance_df["feature"]) == set(val_env.feature_columns)
    assert importance_df["mean_action_shift"].apply(np.isfinite).all()
    assert importance_df["std_action_shift"].apply(np.isfinite).all()
    assert (importance_df["mean_action_shift"] >= 0).all()
    # Sorted descending by mean_action_shift.
    assert list(importance_df["mean_action_shift"]) == sorted(importance_df["mean_action_shift"], reverse=True)
    assert np.isfinite(baseline_metrics["Sharpe Ratio"])


def test_plot_feature_importance_writes_a_nonempty_png(tiny_asset, tmp_path):
    symbol, features_path, price_path = tiny_asset
    model = _train_tiny_ppo(symbol, features_path, price_path)
    val_env = make_env(symbol, features_path, price_path)

    importance_df, _ = permutation_importance_for_policy(model, val_env, n_repeats=1, seed=2)
    out_path = tmp_path / "figs" / "importance.png"
    result_path = plot_feature_importance(importance_df, symbol, out_path)

    assert result_path.exists()
    assert result_path.stat().st_size > 0
