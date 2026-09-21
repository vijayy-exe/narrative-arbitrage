"""Proof (not just a comment) that src/data_pipeline/features.py has no
look-ahead bias: every feature value at time t must be identical whether or
not rows after t exist in the raw input.

Method: write a small synthetic raw dataset (wikipedia + reddit + trends) to
a temporary data/raw/ tree, run process_asset() on a TRUNCATED version of it,
then run it again on the FULL version (with extra future rows appended), and
assert the features at every timestamp present in the truncated run are
bit-for-bit identical in the full run. This is the actual leakage guarantee
the dev plan requires -- previously features.py only had a comment claiming
this was true, with no test proving it, and one feature actually violated it
(see below).
"""
import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import features


ASSET_SYMBOL = "TESTASSET"
N_HISTORY_DAYS = 20   # days visible to the "truncated" run
N_FUTURE_DAYS = 15    # extra days only visible to the "full" run


def _make_full_raw_data(total_days):
    """Generate ONE raw dataset covering `total_days`. Both the 'truncated'
    and 'full' test runs slice a PREFIX of these same arrays, so the
    overlapping days are guaranteed byte-identical between runs -- the only
    difference is extra rows appended after the truncation point. (An
    earlier version of this test generated truncated/full data with two
    separate seeded-RNG calls of different lengths, which is wrong: drawing
    N vs M random values from a fresh RandomState does NOT give you a common
    prefix for every field once more than one field is drawn sequentially --
    it only matches for the very first field. Slicing one generation avoids
    that trap entirely.)
    """
    dates = pd.date_range("2023-01-01", periods=total_days, freq="D")
    rng = np.random.RandomState(42)

    df_wiki = pd.DataFrame({
        "pageviews": rng.randint(500, 5000, total_days),
        "edit_count": rng.randint(0, 20, total_days),
        "unique_editors": rng.randint(0, 10, total_days),
        "revert_count": rng.randint(0, 5, total_days),
    }, index=dates)
    df_wiki.index.name = "date"

    df_reddit_agg = pd.DataFrame({
        "post_count": rng.randint(0, 5, total_days),
        "comment_count": rng.randint(0, 300, total_days),
        "sentiment": rng.uniform(-1, 1, total_days),
    }, index=dates)
    df_reddit_agg.index.name = "date"

    keywords = ["moon", "rocket", "squeeze", "bearish", "bullish", "dump", "pump", "hold"]
    texts = [" ".join(rng.choice(keywords, size=4)) for _ in range(total_days)]
    df_reddit_raw = pd.DataFrame({
        "date": dates,
        "post_count": 1,
        "comment_count": df_reddit_agg["comment_count"].values,
        "sentiment": df_reddit_agg["sentiment"].values,
        "text": texts,
    })

    # Weekly search interest, like real Google Trends output.
    weekly_dates = dates[::7]
    df_trends = pd.DataFrame({
        "search_interest": rng.randint(0, 100, len(weekly_dates)),
    }, index=weekly_dates)
    df_trends.index.name = "date"

    # Deliberate outlier planted strictly in the FUTURE region (after
    # N_HISTORY_DAYS): a huge edit_count spike large enough to blow past any
    # min/max range established by the historical window. This makes the
    # test actually discriminate between the causal (expanding-window) and
    # the old buggy (whole-series) min/max normalization used by
    # sentiment_divergence -- with only mild random data, both
    # implementations can accidentally agree by chance if the random tail
    # never exceeds the historical range. (Verified: without this outlier,
    # the leakage test below silently passed even with the old
    # `vel.min()`/`vel.max()` global-normalization bug reintroduced, purely
    # by luck of the RNG seed -- a false negative. With the outlier planted,
    # reintroducing that bug reliably fails this test, as it should.)
    outlier_idx = min(N_HISTORY_DAYS + 5, total_days - 1)
    if outlier_idx >= N_HISTORY_DAYS:
        df_wiki.iloc[outlier_idx, df_wiki.columns.get_loc("edit_count")] = 10_000

    return dates, df_wiki, df_reddit_agg, df_reddit_raw, df_trends


def _write_raw_tree(tmp_path, dates, df_wiki, df_reddit_agg, df_reddit_raw, df_trends, n_days):
    """Write the first `n_days` days of the given (already-generated) frames."""
    cutoff = dates[n_days - 1]

    for sub in ["wikipedia", "reddit", "trends"]:
        (tmp_path / "data" / "raw" / sub).mkdir(parents=True, exist_ok=True)

    df_wiki.loc[:cutoff].to_csv(tmp_path / "data/raw/wikipedia" / f"{ASSET_SYMBOL}.csv")
    df_reddit_agg.loc[:cutoff].to_csv(tmp_path / "data/raw/reddit" / f"{ASSET_SYMBOL}.csv")
    df_reddit_raw[df_reddit_raw["date"] <= cutoff].to_csv(
        tmp_path / "data/raw/reddit" / f"{ASSET_SYMBOL}_raw.csv", index=False
    )
    df_trends.loc[:cutoff].to_csv(tmp_path / "data/raw/trends" / f"{ASSET_SYMBOL}.csv")


@pytest.fixture
def truncated_and_full_features(tmp_path, monkeypatch):
    """Runs process_asset() twice from two different CWDs (truncated vs full
    raw data) and returns (df_truncated, df_full)."""
    truncated_dir = tmp_path / "truncated"
    full_dir = tmp_path / "full"
    truncated_dir.mkdir()
    full_dir.mkdir()

    total_days = N_HISTORY_DAYS + N_FUTURE_DAYS
    dates, df_wiki, df_reddit_agg, df_reddit_raw, df_trends = _make_full_raw_data(total_days)

    _write_raw_tree(truncated_dir, dates, df_wiki, df_reddit_agg, df_reddit_raw, df_trends, N_HISTORY_DAYS)
    _write_raw_tree(full_dir, dates, df_wiki, df_reddit_agg, df_reddit_raw, df_trends, total_days)

    monkeypatch.chdir(truncated_dir)
    df_truncated = features.process_asset(ASSET_SYMBOL, "Test Asset", assets_config={})

    monkeypatch.chdir(full_dir)
    df_full = features.process_asset(ASSET_SYMBOL, "Test Asset", assets_config={})

    return df_truncated, df_full


def test_features_identical_for_original_rows_when_future_rows_added(truncated_and_full_features):
    df_truncated, df_full = truncated_and_full_features

    assert df_truncated is not None and df_full is not None
    assert not df_truncated.empty

    common_index = df_truncated.index
    assert common_index.isin(df_full.index).all()

    df_full_aligned = df_full.loc[common_index, df_truncated.columns]

    # Compare column by column with an explicit failure message naming the
    # offending column, rather than one opaque assert_frame_equal failure.
    for col in df_truncated.columns:
        pd.testing.assert_series_equal(
            df_truncated[col],
            df_full_aligned[col],
            check_names=False,
            obj=col,
            atol=0, rtol=0,
        )


def test_sentiment_divergence_column_is_covered_by_leakage_check(truncated_and_full_features):
    """Guard against silently dropping the column from the leak check: fail
    loudly if the feature pipeline stops producing sentiment_divergence
    (would make the test above vacuous for that column)."""
    df_truncated, _ = truncated_and_full_features
    assert "sentiment_divergence" in df_truncated.columns
    assert "novelty_score" in df_truncated.columns
