"""Tests for src/env/multi_asset_env.py (Upgrade 5 -- joint multi-asset
portfolio env). Mirrors tests/env/test_env.py's conventions: synthetic
per-asset CSVs written to tmp_path, one test per invariant.
"""
import numpy as np
import pandas as pd
import pytest

from src.env.multi_asset_env import MultiAssetTradingEnv

N_DAYS = 60


def _write_asset_csvs(tmp_path, symbol, seed, n_days=N_DAYS, start="2023-01-01"):
    dates = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.RandomState(seed)

    features_path = tmp_path / f"{symbol}_features.csv"
    price_path = tmp_path / f"{symbol}_price.csv"

    df_features = pd.DataFrame(rng.normal(size=(n_days, 3)), index=dates, columns=["a", "b", "c"])
    df_features.index.name = "date"
    df_features.to_csv(features_path)

    df_price = pd.DataFrame({"Close": 100 + np.cumsum(rng.normal(size=n_days))}, index=dates)
    df_price.index.name = "date"
    df_price.to_csv(price_path)

    return features_path, price_path


def _make_env(tmp_path, symbols=("A", "B", "C"), **kwargs):
    asset_paths = {}
    for i, s in enumerate(symbols):
        asset_paths[s] = _write_asset_csvs(tmp_path, s, seed=i)
    return MultiAssetTradingEnv(asset_paths, **kwargs)


def test_observation_and_action_dims_scale_with_n_assets(tmp_path):
    env = _make_env(tmp_path, symbols=("A", "B", "C"))
    assert env.action_space.shape == (3,)
    # 3 features/asset x 3 assets + 2*3 context (position, pnl per asset) + 1 cash
    assert env.observation_space.shape == (3 * 3 + 2 * 3 + 1,)


def test_reset_and_step_produce_finite_values(tmp_path):
    env = _make_env(tmp_path)
    obs, info = env.reset()
    assert np.isfinite(obs).all()

    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert set(info["positions"].keys()) == {"A", "B", "C"}


def test_do_nothing_policy_yields_zero_reward(tmp_path):
    env = _make_env(tmp_path)
    env.reset()
    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        assert reward == pytest.approx(0.0, abs=1e-9)


def test_date_intersection_bounded_by_shortest_or_latest_starting_asset(tmp_path):
    """One asset starts 10 days later than the other two -> the common
    index must start at the LATER date, dropping the first 10 days for
    every asset, not just the late one."""
    asset_paths = {
        "EARLY1": _write_asset_csvs(tmp_path, "EARLY1", seed=0, start="2023-01-01"),
        "EARLY2": _write_asset_csvs(tmp_path, "EARLY2", seed=1, start="2023-01-01"),
        "LATE": _write_asset_csvs(tmp_path, "LATE", seed=2, start="2023-01-11"),
    }
    env = MultiAssetTradingEnv(asset_paths)
    assert env.dates.min() == pd.Timestamp("2023-01-11")


def test_single_asset_notional_matches_narrative_trading_env_convention(tmp_path):
    """A single-asset MultiAssetTradingEnv (N=1) stepping a full +1 position
    into a known price move should realize the SAME dollar P&L formula as
    NarrativeTradingEnv (position * pct_change * initial_cash) -- i.e. this
    env's per-asset accounting is a direct, unscaled reuse of the already-
    validated single-asset formula, not a new one."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    features_path = tmp_path / "X_features.csv"
    price_path = tmp_path / "X_price.csv"
    pd.DataFrame({"a": [0.0] * 5}, index=dates).rename_axis("date").to_csv(features_path)
    prices = [100.0, 110.0, 110.0, 110.0, 110.0]  # +10% on day 1 -> day 2
    pd.DataFrame({"Close": prices}, index=dates).rename_axis("date").to_csv(price_path)

    env = MultiAssetTradingEnv({"X": (features_path, price_path)}, transaction_fee=0.0)
    env.reset()
    env.step(np.array([1.0], dtype=np.float32))  # open full long at day0's price
    obs, reward, done, truncated, info = env.step(np.array([1.0], dtype=np.float32))  # realize the +10% move
    assert env.cash == pytest.approx(env.initial_cash * 1.10)
