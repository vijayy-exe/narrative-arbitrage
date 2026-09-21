import os
import warnings
import yaml
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

def load_config():
    with open("configs/assets.yaml", "r") as f:
        assets = yaml.safe_load(f)
    return assets

def calculate_derivatives(series):
    velocity = series.diff()
    acceleration = velocity.diff()
    return velocity, acceleration


# Signals that get <signal>_velocity/_acceleration columns. Shared between
# process_asset() (computed once on the full calendar-day grid) and
# recompute_derivatives_on_index() (recomputed on a consumer's own,
# possibly-reduced index -- see that function's docstring for why both
# exist).
DERIVATIVE_SIGNALS = [
    'pageviews', 'edit_count', 'comment_count', 'search_interest',
    'hn_story_count', 'hn_points', 'hn_comment_count',
]


def _apply_derivatives(df, signals=DERIVATIVE_SIGNALS):
    df = df.copy()
    for sig in signals:
        if sig in df.columns:
            vel, acc = calculate_derivatives(df[sig])
            df[f"{sig}_velocity"] = vel
            df[f"{sig}_acceleration"] = acc
    return df


def _apply_sentiment_divergence(df):
    if 'sentiment' not in df.columns or 'edit_count_velocity' not in df.columns:
        return df
    df = df.copy()
    vel = df['edit_count_velocity']
    running_min = vel.expanding().min()
    running_max = vel.expanding().max()
    norm_vel = (vel - running_min) / (running_max - running_min + 1e-6)
    df['sentiment_divergence'] = df['sentiment'] - norm_vel
    return df


def compute_edit_war_ratio(df):
    """edit_war_ratio = revert_count / (edit_count + 1) -- see
    docs/feature_dictionary.md for the exact rationale (the `+1` avoids
    divide-by-zero on no-edit days). Factored out of process_asset() as its
    own function (2026-09, Phase 7) so src/realtime/scheduler.py can compute
    the identical formula for a live daily row instead of reimplementing it
    inline -- the project's stated policy of never re-deriving feature logic
    outside src/data_pipeline/features.py.
    """
    if 'edit_count' not in df.columns or 'revert_count' not in df.columns:
        return df
    df = df.copy()
    df['edit_war_ratio'] = df['revert_count'] / (df['edit_count'] + 1)
    return df


def recompute_derivatives_on_index(df):
    """Recompute <signal>_velocity/_acceleration and sentiment_divergence on
    `df`'s OWN index, from the level columns -- for use by any consumer
    (NarrativeTradingEnv, src/baselines/supervised.py,
    src/eval/walk_forward.py's evaluate_supervised_roll) AFTER it joins the
    processed features CSV against a price index, in place of trusting
    process_asset()'s calendar-grid-computed derivative columns.

    WHY THIS EXISTS (bug found 2026-09-07, while investigating why PPO and
    the supervised baseline both underperformed momentum specifically on
    GME/AMC): process_asset() computes velocity/acceleration once on the
    full CALENDAR-DAY grid (7 days/week, via ffill for search_interest).
    Google Trends reports weekly, anchored to a FIXED day (every raw
    timestamp in data/raw/trends/*.csv is a Sunday) -- so the ffilled
    search_interest level only ever CHANGES on a Sunday, meaning
    search_interest_velocity/_acceleration on the calendar grid are nonzero
    ONLY on Sundays. Stock assets (GME, AMC; priced via yfinance) only have
    price rows on weekdays. Every consumer that inner-joins the calendar-
    grid features against a stock's weekday-only price index was therefore
    silently discarding 100% of search_interest's real signal for GME/AMC --
    confirmed live: search_interest_velocity was a CONSTANT ZERO across all
    355 of GME's price-aligned rows, despite std=5.0 over the full 518-row
    calendar grid. This wasn't just a missed opportunity: search_interest_
    velocity is the single strongest cross-asset predictor found in this
    project (correlation with next-day return: 0.23 for DOGE, 0.38 for
    SHIB) on the assets where it survives the join (crypto, priced 7
    days/week via ccxt, so this bug never applied there) -- and completely
    absent from GME/AMC's observation space in practice, not by design.

    Fix: recompute the derivative fresh on each consumer's OWN
    (price-joined) index, from the level columns, AFTER that join. This
    makes "velocity" mean "change since this asset's last actual trading
    day" -- the economically correct question for a same-day trading
    decision anyway, not "change since yesterday on a calendar this asset
    doesn't trade on." Still strictly causal/leak-free: diff() at row t only
    reads row t-1 of the SAME (already-reduced) index, same guarantee
    process_asset() itself relies on (see tests/data_pipeline/
    test_no_leakage.py, which does not cover this post-join step since it
    never joins price -- this function has no separate leakage test because
    it is a pure per-row .diff()/.expanding() recomputation of already-
    leakage-tested logic, just on a different index).
    """
    df = _apply_derivatives(df)
    df = _apply_sentiment_divergence(df)
    return df

def calculate_novelty_score(texts_series, window=7):
    # texts_series is a series where index is date and value is a string of all text for that day
    novelty = []
    dates = texts_series.index

    vectorizer = TfidfVectorizer(stop_words='english')

    for i in range(len(dates)):
        if i < window:
            novelty.append(np.nan)
            continue

        current_text = [texts_series.iloc[i]]
        past_texts = texts_series.iloc[i-window:i].tolist()

        try:
            all_texts = current_text + past_texts
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            current_vec = tfidf_matrix[0]
            past_vecs = tfidf_matrix[1:]

            # Average past vector
            avg_past_vec = np.mean(past_vecs.toarray(), axis=0).reshape(1, -1)

            sim = cosine_similarity(current_vec, avg_past_vec)[0][0]
            novelty.append(1 - sim) # Distance
        except ValueError:
            # Expected/benign: TfidfVectorizer raises ValueError("empty vocabulary...")
            # when every document in this window is empty or contains only
            # stop words (e.g. no Reddit text at all on quiet days). NaN is
            # the correct output for "novelty is undefined here", not a bug.
            novelty.append(np.nan)
        except Exception as e:
            # Anything else is unexpected -- warn loudly instead of silently
            # hiding a real bug behind a bare except (the previous behavior).
            warnings.warn(
                f"Unexpected error computing novelty score at {dates[i]}: "
                f"{type(e).__name__}: {e}"
            )
            novelty.append(np.nan)

    return pd.Series(novelty, index=dates)

def process_asset(asset_symbol, asset_name, assets_config):
    print(f"Processing features for {asset_symbol}...")

    # Paths. NOTE: price is intentionally never read here -- price is kept
    # strictly separate from attention features (see price.py docstring /
    # dev plan Phase 1a) and is only ever used later, by reward/backtest code.
    raw_wiki_path = Path(f"data/raw/wikipedia/{asset_symbol}.csv")
    raw_reddit_path = Path(f"data/raw/reddit/{asset_symbol}.csv")
    raw_trends_path = Path(f"data/raw/trends/{asset_symbol}.csv")
    raw_hn_path = Path(f"data/raw/hackernews/{asset_symbol}.csv")

    # Reddit ingestion requires live API credentials (REDDIT_CLIENT_ID/
    # SECRET/USER_AGENT) that are frequently not configured (see reddit.py) --
    # don't hard-require it. As of 2026-09, Reddit no longer even issues
    # those credentials self-serve (see configs/assets.yaml), so this is a
    # documented, currently-blocked gap, not just an unconfigured-env case.
    # Wikipedia and Trends need no credentials, so we require at least one of
    # those two to exist; Reddit-derived columns (comment_count*, sentiment*,
    # novelty_score, sentiment_divergence) simply won't be present in the
    # output when Reddit data is missing, and this is loud (`print`) about
    # which sources went into each asset. Hacker News (src/data_pipeline/
    # hackernews.py) is the real, no-auth forum-attention signal used
    # instead -- see that module's docstring for why it replaced Reddit.
    available = {
        "wikipedia": raw_wiki_path.exists(),
        "reddit": raw_reddit_path.exists(),
        "trends": raw_trends_path.exists(),
        "hackernews": raw_hn_path.exists(),
    }
    if not (available["wikipedia"] or available["trends"]):
        print(f"Missing raw data for {asset_symbol} (no wikipedia or trends data found). Skipping.")
        return None
    missing = [src for src, ok in available.items() if not ok]
    if missing:
        print(f"{asset_symbol}: proceeding without {missing} (not fetched -- see caveats in reddit.py/hackernews.py).")

    # Load whatever sources are available.
    frames = []
    if available["wikipedia"]:
        frames.append(pd.read_csv(raw_wiki_path, index_col='date', parse_dates=True))
    if available["reddit"]:
        frames.append(pd.read_csv(raw_reddit_path, index_col='date', parse_dates=True))
    if available["trends"]:
        df_trends = pd.read_csv(raw_trends_path, index_col='date', parse_dates=True)
        frames.append(df_trends)
    if available["hackernews"]:
        df_hn = pd.read_csv(raw_hn_path, index_col='date', parse_dates=True)
        frames.append(df_hn)

    df = frames[0].join(frames[1:], how='outer').sort_index() if len(frames) > 1 else frames[0].sort_index()

    # Missing-data handling (documented, see docs/feature_dictionary.md):
    # - search_interest (Google Trends) is reported WEEKLY even though our
    #   bars are daily. Zero-filling it would fabricate a sawtooth (real
    #   value on the report day, then a fake collapse to 0 for six days,
    #   then a fake vertical velocity/acceleration spike every single week)
    #   -- that's a bigger distortion than the missingness itself. We
    #   forward-fill it instead (the last *known, past* reading is the best
    #   causal estimate until the next report), which is still leak-free
    #   since ffill only ever propagates a value forward in time, never
    #   backward.
    # - Everything else (pageviews, edit_count, comment_count, sentiment,
    #   hn_story_count, hn_points, hn_comment_count, etc.) is zero-filled: a
    #   missing daily row from Wikipedia/Reddit/Hacker News genuinely means
    #   "nothing happened that day" (or "not fetched"), not "unknown", so 0
    #   is the correct causal fill.
    if 'search_interest' in df.columns:
        df['search_interest'] = df['search_interest'].ffill()
    df = df.fillna(0)

    # 1. Derivatives (on the full calendar-day grid -- see
    # recompute_derivatives_on_index()'s docstring for why consumers that
    # join this file against a weekday-only price index must NOT just trust
    # these columns as-is, and instead recompute them post-join).
    df = _apply_derivatives(df)

    # 2. Edit-war ratio
    df = compute_edit_war_ratio(df)

    # 3. Novelty Score
    # We need the original text from reddit.py
    # We read from the raw file which contains the text
    raw_reddit_text_path = Path(f"data/raw/reddit/{asset_symbol}_raw.csv")
    if raw_reddit_text_path.exists():
        df_reddit_raw = pd.read_csv(raw_reddit_text_path, parse_dates=['date'])
        df_reddit_raw = df_reddit_raw.set_index('date').sort_index()

        # Group texts by date in case there are multiple posts per day
        texts_series = df_reddit_raw.groupby('date')['text'].apply(lambda x: " ".join(x))
        df['novelty_score'] = calculate_novelty_score(texts_series)

    # 4. Cross-platform sentiment divergence: Reddit sentiment - normalized
    # Wikipedia edit velocity, via _apply_sentiment_divergence() (uses an
    # *expanding* causal min/max, not the whole-series min/max the original
    # implementation used -- see recompute_derivatives_on_index()'s
    # docstring / that fix's history for why the causal version is
    # required for leak-freeness).
    df = _apply_sentiment_divergence(df)

    # CRITICAL: No look-ahead bias.
    # All transforms above (.diff() velocity/acceleration, the trailing
    # `window`-day slice in calculate_novelty_score, and the elementwise
    # edit-war-ratio/sentiment-divergence arithmetic) only ever read rows at
    # or before the current index -- none of them reach forward. This is
    # actually verified, not just asserted in a comment: see
    # tests/data_pipeline/test_no_leakage.py, which computes features on a
    # truncated dataset, recomputes after appending additional future rows,
    # and asserts every feature value at the original timestamps is
    # bit-for-bit identical. Run it whenever this module changes.

    return df

def main():
    assets = load_config()
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Flatten assets
    all_assets = []
    for category in assets.values():
        for asset in category:
            all_assets.append(asset)

    for asset in all_assets:
        symbol = asset.get('ticker') or asset.get('symbol') or asset.get('id')
        name = asset.get('name')

        df_features = process_asset(symbol, name, assets)
        if df_features is not None:
            df_features.to_csv(processed_dir / f"{symbol}_features.csv")
            print(f"Saved features for {symbol}")

if __name__ == "__main__":
    main()
