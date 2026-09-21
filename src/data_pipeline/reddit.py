"""Reddit attention-signal fetcher.

Fix history (see narrative-arbitrage-dev-plan Phase 1a review):
  The previous implementation did NOT use PRAW at all -- it hit the
  unauthenticated `https://www.reddit.com/search.json` endpoint with a
  browser-spoofed User-Agent and searched **globally** ("all of Reddit") for
  the bare asset name, then filtered results to the target date range on the
  client side. As of 2026-09 this endpoint returns HTTP 403 for
  unauthenticated requests (verified live) -- Reddit requires OAuth for API
  access now, so that approach cannot fetch any real data any more, live or
  otherwise. It also violated the dev plan's explicit request for "relevant
  subreddits" (plural, specific), not a global search.

  This version uses PRAW with a real (read-only) OAuth client, restricted to
  a per-asset list of genuinely relevant subreddits (`subreddits:` in
  configs/assets.yaml), via `reddit.subreddit("+".join(subreddits))`.

KNOWN DATA-COMPLETENESS CAVEAT (please read before trusting this as a
complete daily time series):
  PRAW's `.search()` (like all of Reddit's public search) is capped at
  roughly 1000 results per query, and is relevance-ranked rather than
  date-complete. For a ~17-month backfill window (configs/date_range.yaml),
  1000 results is very unlikely to cover every day uniformly, especially for
  high-volume subreddits/assets -- expect gaps, especially early in the
  window relative to when the query is run. Pushshift (which used to offer
  genuinely complete historical search) has long been access-restricted and
  is not used here. This is a real, structural limitation of PRAW-only
  ingestion for a long historical backfill, not a bug to "fix" -- if
  complete historical daily coverage is required, use an approved
  Reddit data licensing / Pushshift-successor product instead. This module
  is honest about that gap rather than presenting a sample as if it were a
  complete series.
"""

import os
import time
import yaml
import pandas as pd
import praw
import nltk
from pathlib import Path
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

load_dotenv()

# Download VADER lexicon if not already present (no-op, no network hit, if cached)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


def load_config():
    with open("configs/assets.yaml", "r") as f:
        assets = yaml.safe_load(f)
    with open("configs/date_range.yaml", "r") as f:
        dates = yaml.safe_load(f)
    return assets, dates


def iter_assets(assets):
    """Flatten configs/assets.yaml into asset dicts with a normalized 'symbol' key."""
    for category in assets.values():
        for asset in category:
            symbol = asset.get("ticker") or asset.get("symbol") or asset.get("id")
            yield {**asset, "symbol": symbol}


def get_reddit_client():
    """Build a read-only PRAW client from environment variables (loaded from
    .env via python-dotenv, see .env.example). Returns None if credentials
    are not configured, rather than raising -- callers should handle that by
    skipping Reddit ingestion and reporting it plainly, not crashing the
    whole pipeline run.
    """
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    if not all([client_id, client_secret, user_agent]):
        return None

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    reddit.read_only = True
    return reddit


def fetch_reddit_data(reddit, asset_name, subreddits, start_date, end_date, limit=1000):
    """Search `subreddits` (list of subreddit names) for `asset_name` and
    return one row per matching submission within [start_date, end_date].

    See module docstring for the ~1000-result search cap / date-completeness
    caveat -- this is NOT guaranteed to be a complete daily series.
    """
    print(f"Fetching Reddit data for {asset_name} from r/{'+'.join(subreddits)}...")
    sia = SentimentIntensityAnalyzer()

    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()

    subreddit = reddit.subreddit("+".join(subreddits))

    all_posts = []
    try:
        for submission in subreddit.search(asset_name, sort="new", time_filter="all", limit=limit):
            post_date = pd.to_datetime(submission.created_utc, unit="s").date()
            if not (start_dt <= post_date <= end_dt):
                continue
            text = f"{submission.title} {submission.selftext or ''}"
            sentiment = sia.polarity_scores(text)["compound"]
            all_posts.append({
                "date": post_date,
                "post_count": 1,
                "comment_count": submission.num_comments,
                "sentiment": sentiment,
                "text": text,
            })
    except Exception as e:
        # praw/prawcore exceptions (rate limits, auth errors, banned/private
        # subreddit, etc.) -- log and return whatever we gathered so far
        # rather than crashing the whole ingestion run.
        print(f"Error fetching Reddit data for {asset_name} from r/{'+'.join(subreddits)}: {e}")

    return pd.DataFrame(all_posts)


def main():
    assets, dates = load_config()
    start = dates["start_date"]
    end = dates["end_date"]

    raw_dir = Path("data/raw/reddit")
    raw_dir.mkdir(parents=True, exist_ok=True)

    reddit = get_reddit_client()
    if reddit is None:
        print(
            "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT are not "
            "set (checked environment + .env). Skipping live Reddit ingestion. "
            "See .env.example for how to obtain free Reddit API credentials."
        )
        return

    for asset in iter_assets(assets):
        symbol = asset["symbol"]
        name = asset["name"]
        subreddits = asset.get("subreddits")
        if not subreddits:
            print(f"No 'subreddits' configured for {symbol} in configs/assets.yaml. Skipping.")
            continue

        raw_file_path = raw_dir / f"{symbol}_raw.csv"
        agg_file_path = raw_dir / f"{symbol}.csv"

        if raw_file_path.exists():
            print(f"Reddit data for {symbol} already exists. Skipping.")
            continue

        df_raw = fetch_reddit_data(reddit, name, subreddits, start, end)
        if df_raw.empty:
            print(f"No Reddit data found for {symbol} ({name}) in r/{'+'.join(subreddits)}")
            continue

        df_raw.to_csv(raw_file_path, index=False)

        daily = df_raw.groupby("date").agg({
            "post_count": "sum",
            "comment_count": "sum",
            "sentiment": "mean",
        })
        daily.index.name = "date"
        daily.to_csv(agg_file_path)
        print(f"Saved Reddit data for {symbol} -> {agg_file_path} ({len(df_raw)} matching submissions)")

        time.sleep(2)  # be polite / avoid rate limits between assets


if __name__ == "__main__":
    main()
