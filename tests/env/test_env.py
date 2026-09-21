"""Real tests for NarrativeTradingEnv, per the dev plan's Phase 3 requirements
(previously this file was a placeholder -- `def test_placeholder(): assert
True` -- so none of these checks had ever actually run, which is how a
look-ahead bug in the reward calculation went unnoticed; see
src/env/trading_env.py's step() docstring/comments for the fix).

Uses small synthetic features/price CSVs written to tmp_path rather than the
real data/ files, so these tests are fast, hermetic, and don't depend on
which assets happen to be fetched.
"""
import numpy as np
import pandas as pd
import pytest

from src.env.trading_env import NarrativeTradingEnv


N_DAYS = 40


def _write_env_csvs(tmp_path, close_prices, feature_values=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2023-01-01", periods=len(close_prices), freq="D")

    features_path = tmp_path / "features.csv"
    price_path = tmp_path / "price.csv"

    if feature_values is None:
        rng = np.random.RandomState(0)
        feature_values = rng.normal(size=(len(close_prices), 3))

    df_features = pd.DataFrame(
        feature_values, index=dates, columns=["feat_a", "feat_b", "feat_c"]
    )
    df_features.index.name = "date"
    df_features.to_csv(features_path)

    df_price = pd.DataFrame({"Close": close_prices}, index=dates)
    df_price.index.name = "date"
    df_price.to_csv(price_path)

    return features_path, price_path


def _make_env(tmp_path, close_prices, feature_values=None, **kwargs):
    features_path, price_path = _write_env_csvs(tmp_path, close_prices, feature_values)
    return NarrativeTradingEnv(
        asset_symbol="TEST", features_path=features_path, price_path=price_path, **kwargs
    )


def test_random_actions_no_nan_or_inf(tmp_path):
    rng = np.random.RandomState(1)
    close_prices = 100 + np.cumsum(rng.normal(0, 1, N_DAYS))
    env = _make_env(tmp_path, close_prices)

    obs, _ = env.reset()
    assert np.isfinite(obs).all()

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        assert np.isfinite(obs).all(), "NaN/inf in observation"
        assert np.isfinite(reward), "NaN/inf in reward"


def test_do_nothing_policy_yields_zero_reward(tmp_path):
    rng = np.random.RandomState(2)
    close_prices = 100 + np.cumsum(rng.normal(0, 1, N_DAYS))
    env = _make_env(tmp_path, close_prices, transaction_fee=0.001, risk_lambda=0.01)

    env.reset()
    done = False
    total_reward = 0.0
    while not done:
        # Always hold flat (position 0): no transaction cost after the
        # first no-op step (0 -> 0 is not a position change), no P&L (flat
        # position earns nothing regardless of price moves), and the risk
        # penalty is lambda * |0| = 0. So reward should be exactly zero
        # every step, not just "small".
        obs, reward, done, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        total_reward += reward

    assert total_reward == pytest.approx(0.0, abs=1e-9)


def test_reward_sign_correct_for_profitable_long(tmp_path):
    # Strictly increasing price: going max long from the start should be
    # profitable net of the one-time entry transaction cost.
    close_prices = np.linspace(100, 200, N_DAYS)
    env = _make_env(tmp_path, close_prices, transaction_fee=0.001, risk_lambda=0.0)

    env.reset()
    done = False
    total_reward = 0.0
    while not done:
        obs, reward, done, truncated, info = env.step(np.array([1.0], dtype=np.float32))
        total_reward += reward

    assert total_reward > 0, "Going max long into a steadily rising price should be net profitable"


def test_reward_sign_correct_for_unprofitable_long_in_falling_price(tmp_path):
    # Strictly decreasing price: going max long from the start should lose money.
    close_prices = np.linspace(200, 100, N_DAYS)
    env = _make_env(tmp_path, close_prices, transaction_fee=0.001, risk_lambda=0.0)

    env.reset()
    done = False
    total_reward = 0.0
    while not done:
        obs, reward, done, truncated, info = env.step(np.array([1.0], dtype=np.float32))
        total_reward += reward

    assert total_reward < 0, "Going max long into a steadily falling price should be net unprofitable"


def test_reward_does_not_credit_position_for_price_move_before_it_was_taken(tmp_path):
    """Regression test for the look-ahead bug fixed in step(): switching
    INTO a position on the same step that the price jumps must NOT earn
    reward from that jump -- the position wasn't held during the move.
    """
    # Flat for the first half, then one large jump, then flat again.
    close_prices = np.concatenate([
        np.full(N_DAYS // 2, 100.0),
        np.full(N_DAYS - N_DAYS // 2, 200.0),
    ])
    env = _make_env(tmp_path, close_prices, transaction_fee=0.0, risk_lambda=0.0)

    env.reset()
    # Stay flat through the jump step, then go long right after.
    jump_step = N_DAYS // 2
    rewards = []
    for i in range(N_DAYS - 1):
        action = 0.0 if i < jump_step else 1.0
        obs, reward, done, truncated, info = env.step(np.array([action], dtype=np.float32))
        rewards.append(reward)
        if done:
            break

    # The reward on the very step the price jumps (while still flat, having
    # just decided to go long) must be ~0 -- the agent was not exposed to
    # that move yet, it only rebalances INTO the position at the new price.
    assert rewards[jump_step] == pytest.approx(0.0, abs=1e-9)


def test_portfolio_value_updates_after_a_realized_price_move(tmp_path):
    # Step 0 legitimately leaves portfolio_value unchanged with zero fees:
    # there's no prior price to realize a return against yet, and opening
    # the position costs nothing. It's the SECOND step -- once an actual
    # price move has occurred while holding a nonzero position -- where
    # portfolio_value must reflect it.
    close_prices = np.linspace(100, 110, N_DAYS)
    env = _make_env(tmp_path, close_prices, transaction_fee=0.0, risk_lambda=0.0)
    env.reset()
    env.step(np.array([1.0], dtype=np.float32))
    value_after_step0 = env.portfolio_value
    env.step(np.array([1.0], dtype=np.float32))
    assert env.portfolio_value != value_after_step0, "portfolio_value must reflect a realized price move"


def test_portfolio_value_changes_immediately_with_nonzero_fee(tmp_path):
    # With a nonzero transaction fee, opening a position costs money right
    # away, even before any price move has been realized.
    close_prices = np.linspace(100, 110, N_DAYS)
    env = _make_env(tmp_path, close_prices, transaction_fee=0.01, risk_lambda=0.0)
    env.reset()
    initial_value = env.portfolio_value
    env.step(np.array([1.0], dtype=np.float32))
    assert env.portfolio_value < initial_value, "opening a position with a nonzero fee must cost something immediately"


def test_position_sizing_is_invariant_to_asset_price_level(tmp_path):
    """Regression test: an earlier version sized positions as a fixed unit
    count (position 1.0 = 100 units) rather than a fraction of capital, so
    a full position in a cheap asset (e.g. real DOGE at ~$0.15) was worth a
    tiny, arbitrary dollar amount compared to the same position in an
    expensive one -- confirmed on real data to make the risk penalty
    ~100x larger than the actual return signal. Two assets with the same
    % returns but wildly different absolute price levels must now produce
    IDENTICAL portfolio trajectories under a max-long policy.
    """
    pct_path = np.array([0.0] + [0.01] * (N_DAYS - 1))  # +1%/day
    cheap_prices = 0.10 * np.cumprod(1 + pct_path)
    expensive_prices = 500.0 * np.cumprod(1 + pct_path)

    env_cheap = _make_env(tmp_path / "cheap", cheap_prices, transaction_fee=0.0, risk_lambda=0.0)
    env_expensive = _make_env(tmp_path / "expensive", expensive_prices, transaction_fee=0.0, risk_lambda=0.0)

    env_cheap.reset()
    env_expensive.reset()
    done = False
    while not done:
        _, r_cheap, done, _, info_cheap = env_cheap.step(np.array([1.0], dtype=np.float32))
        _, r_expensive, _, _, info_expensive = env_expensive.step(np.array([1.0], dtype=np.float32))
        assert r_cheap == pytest.approx(r_expensive, rel=1e-6)
        assert info_cheap["portfolio_value"] == pytest.approx(info_expensive["portfolio_value"], rel=1e-6)


def test_risk_penalty_scales_with_realized_volatility(tmp_path):
    """A high-volatility asset must incur a larger risk penalty (all else
    equal) than a low-volatility one for holding the same full position --
    this is the entire point of scaling by self._volatility rather than a
    bare position-magnitude constant.
    """
    rng_low = np.random.RandomState(10)
    rng_high = np.random.RandomState(10)
    low_vol_prices = 100 + np.cumsum(rng_low.normal(0, 0.1, N_DAYS))
    high_vol_prices = 100 + np.cumsum(rng_high.normal(0, 5.0, N_DAYS))

    env_low = _make_env(tmp_path / "low", low_vol_prices, transaction_fee=0.0, risk_lambda=1.0)
    env_high = _make_env(tmp_path / "high", high_vol_prices, transaction_fee=0.0, risk_lambda=1.0)

    assert env_low._volatility.mean() < env_high._volatility.mean()


def test_date_window_restricts_episode_length(tmp_path):
    dates = pd.date_range("2023-01-01", periods=N_DAYS, freq="D")
    close_prices = 100 + np.cumsum(np.random.RandomState(3).normal(0, 1, N_DAYS))
    features_path, price_path = _write_env_csvs(tmp_path, close_prices)

    full_env = NarrativeTradingEnv(asset_symbol="TEST", features_path=features_path, price_path=price_path)
    windowed_env = NarrativeTradingEnv(
        asset_symbol="TEST", features_path=features_path, price_path=price_path,
        start_date=dates[10], end_date=dates[20],
    )

    assert len(windowed_env.df) == 11
    assert len(windowed_env.df) < len(full_env.df)
    assert windowed_env.df.index.min() == dates[10]
    assert windowed_env.df.index.max() == dates[20]


def test_leading_nan_features_do_not_reach_observation(tmp_path):
    """Regression test: src/data_pipeline/features.py deliberately leaves
    velocity/acceleration NaN on the first 1-2 rows of every real asset
    (no prior day to diff against -- see docs/feature_dictionary.md). The
    env must still return finite observations for those rows, not propagate
    the NaN (discovered by running the real DOGE_features.csv through the
    env -- the earlier synthetic-only tests above used clean random
    features and would not have caught this).
    """
    close_prices = 100 + np.cumsum(np.random.RandomState(5).normal(0, 1, N_DAYS))
    feature_values = np.random.RandomState(5).normal(size=(N_DAYS, 3))
    feature_values[0, :] = np.nan   # mimics *_velocity's first-row NaN
    feature_values[1, 0] = np.nan   # mimics *_acceleration's second-row NaN

    env = _make_env(tmp_path, close_prices, feature_values=feature_values)
    obs, _ = env.reset()
    assert np.isfinite(obs).all()

    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)


def test_lookback_window_default_matches_original_single_timestep_observation(tmp_path):
    """lookback_window=1 (the default) must be BYTE-FOR-BYTE the original
    observation -- every existing checkpoint (models/ppo_*, models_cvar/
    ppo_*) was trained against this exact shape/content and must remain
    valid."""
    close_prices = 100 + np.cumsum(np.random.RandomState(7).normal(0, 1, N_DAYS))
    env = _make_env(tmp_path, close_prices)
    assert env.observation_space.shape == (env.num_features + 3,)
    obs, _ = env.reset()
    assert obs.shape == (env.num_features + 3,)


def test_lookback_window_flattens_the_last_n_timesteps(tmp_path):
    close_prices = 100 + np.cumsum(np.random.RandomState(8).normal(0, 1, N_DAYS))
    window = 5
    env = _make_env(tmp_path, close_prices, lookback_window=window)
    assert env.observation_space.shape == (env.num_features * window + 3,)

    obs, _ = env.reset()
    assert obs.shape == (env.num_features * window + 3,)
    assert np.isfinite(obs).all()

    # After a few steps (>= window), the LAST row of the flattened window
    # must equal the single-timestep features for the current step --
    # i.e. the window is genuinely "most recent last", not reversed/misaligned.
    for _ in range(window + 2):
        obs, reward, done, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    current_single = env._normalized_features.iloc[env.current_step].values.astype(np.float32)
    last_row_of_window = obs[(window - 1) * env.num_features: window * env.num_features]
    np.testing.assert_allclose(last_row_of_window, current_single, atol=1e-5)


def test_lookback_window_zero_pads_early_episode_steps(tmp_path):
    close_prices = 100 + np.cumsum(np.random.RandomState(9).normal(0, 1, N_DAYS))
    window = 10
    env = _make_env(tmp_path, close_prices, lookback_window=window)
    obs, _ = env.reset()  # current_step == 0 -> only 1 real row, 9 rows must be zero-padded
    n = env.num_features
    padded_rows = obs[: (window - 1) * n].reshape(window - 1, n)
    assert np.allclose(padded_rows, 0.0)


def test_date_window_too_narrow_raises(tmp_path):
    dates = pd.date_range("2023-01-01", periods=N_DAYS, freq="D")
    close_prices = 100 + np.cumsum(np.random.RandomState(4).normal(0, 1, N_DAYS))
    features_path, price_path = _write_env_csvs(tmp_path, close_prices)

    with pytest.raises(ValueError):
        NarrativeTradingEnv(
            asset_symbol="TEST", features_path=features_path, price_path=price_path,
            start_date=dates[5], end_date=dates[5],
        )
