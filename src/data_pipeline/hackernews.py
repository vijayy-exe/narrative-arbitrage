"""Hacker News attention-signal fetcher (via the Algolia HN Search API).

Added as a real, no-auth, no-approval-wait replacement for the Reddit forum
signal: as of 2026-09, reddit.com/prefs/apps no longer issues script-app API
credentials self-serve (redirects to a manual "Responsible Builder Policy"
support-ticket review -- see configs/assets.yaml). hn.algolia.com is free,
needs no key, and has a genuine historical archive back to 2006 (verified
live against the project's full 2023-01-01..2024-06-01 window before writing
this module), unlike e.g. StockTwits (its anonymous endpoint is now behind a
Cloudflare JS challenge -- confirmed live, not usable without solving a bot
challenge, which this project does not do) or GDELT (returned HTTP 429
"please limit requests" on every attempt from this environment, including
via a separate fetch path -- not something pacing on our end could fix, so
it was dropped rather than shipped unverified).

QUERY TERM CAVEAT (verified live, 2026-09-07): bare tickers are unusable as
HN search queries -- "AMC" alone matches 29,910 stories (matches "AMC" as a
substring/acronym across unrelated HN content, not the asset) and "GME"
alone matches 40,386. `configs/assets.yaml`'s `hn_query` field (falls back
to `name`) pins a qualified term per asset instead -- see that file's
comment block for the exact verified hit counts.

COVERAGE CAVEAT: HN's total volume for a niche/mid-cap meme-stock or
small-cap-crypto term over a 17-month window is small (single digits to low
hundreds -- verified live, see configs/assets.yaml), so a single
Algolia query per asset (paginated if needed) covers the full window; no
per-day chunking is needed the way it would be for a high-volume term. This
also means the resulting `hn_story_count` signal is real but sparse for most
assets -- most days will genuinely be 0, which is the correct value (no
matching HN story that day), not a fetch failure.
"""

import time
import yaml
import requests
import pandas as pd
from pathlib import Path

BASE_URL = "https://hn.algolia.com/api/v1/search_by_date"
HITS_PER_PAGE = 1000


def load_config():
    with open("configs/assets.yaml", "r") as f:
        assets = yaml.safe_load(f)
    with open("configs/date_range.yaml", "r") as f:
        dates = yaml.safe_load(f)
    return assets, dates


def iter_assets(assets):
    """Flatten configs/assets.yaml into asset dicts with a normalized
    'symbol' key and a resolved 'query' (hn_query if set, else name)."""
    for category in assets.values():
        for asset in category:
            symbol = asset.get("ticker") or asset.get("symbol") or asset.get("id")
            query = asset.get("hn_query") or asset.get("name")
            yield {**asset, "symbol": symbol, "query": query}


def fetch_hn_stories(query, start_date, end_date, max_pages=20):
    """Fetch all HN stories matching `query` with created_at in
    [start_date, end_date] (inclusive), paginating via Algolia's `page`
    param until exhausted. Returns a list of dicts (one per story) with
    date, points, num_comments, title.

    See module docstring for why a single query (rather than per-day
    chunking) is sufficient for this project's asset basket.
    """
    print(f"Fetching Hacker News stories for query={query!r}...")
    start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    end_ts = int(pd.Timestamp(end_date, tz="UTC").timestamp()) + 86399  # inclusive of end_date

    all_hits = []
    for page in range(max_pages):
        params = {
            "query": query,
            "tags": "story",
            "numericFilters": f"created_at_i>{start_ts},created_at_i<{end_ts}",
            "hitsPerPage": HITS_PER_PAGE,
            "page": page,
        }
        for attempt in range(3):
            try:
                response = requests.get(BASE_URL, params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"  Request error (attempt {attempt + 1}/3): {e}")
                time.sleep(3)
                continue
            if response.status_code == 200:
                break
            print(f"  HTTP {response.status_code} (attempt {attempt + 1}/3)")
            time.sleep(3)
        else:
            print(f"  Giving up on page {page} for query={query!r} after 3 failed attempts.")
            break

        data = response.json()
        hits = data.get("hits", [])
        all_hits.extend(hits)

        nb_pages = data.get("nbPages", 1)
        if page + 1 >= nb_pages or not hits:
            if page + 1 >= max_pages:
                print(
                    f"  WARNING: hit max_pages={max_pages} safety limit for query={query!r}; "
                    f"results may be truncated (nbPages={nb_pages})."
                )
            break
        time.sleep(0.2)  # be polite

    print(f"  {len(all_hits)} stories found for query={query!r} in [{start_date}, {end_date}]")

    rows = []
    for hit in all_hits:
        created_at_i = hit.get("created_at_i")
        if created_at_i is None:
            continue
        date = pd.to_datetime(created_at_i, unit="s", utc=True).date()
        rows.append({
            "date": date,
            "points": hit.get("points") or 0,
            "num_comments": hit.get("num_comments") or 0,
            "title": hit.get("title") or "",
            "object_id": hit.get("objectID"),
        })
    return rows


def main():
    assets, dates = load_config()
    start = dates["start_date"]
    end = dates["end_date"]

    raw_dir = Path("data/raw/hackernews")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for asset in iter_assets(assets):
        symbol = asset["symbol"]
        query = asset["query"]

        raw_file_path = raw_dir / f"{symbol}_raw.csv"
        agg_file_path = raw_dir / f"{symbol}.csv"

        if raw_file_path.exists():
            print(f"Hacker News data for {symbol} already exists. Skipping.")
            continue

        rows = fetch_hn_stories(query, start, end)
        df_raw = pd.DataFrame(rows)
        if df_raw.empty:
            print(f"No Hacker News data found for {symbol} (query={query!r})")
            # Still write an empty raw file so re-runs don't keep re-fetching
            # a genuinely-zero-hit query.
            pd.DataFrame(columns=["date", "points", "num_comments", "title", "object_id"]).to_csv(
                raw_file_path, index=False
            )
            continue

        df_raw.to_csv(raw_file_path, index=False)

        daily = df_raw.groupby("date").agg(
            hn_story_count=("object_id", "count"),
            hn_points=("points", "sum"),
            hn_comment_count=("num_comments", "sum"),
        )
        daily.index.name = "date"
        daily.to_csv(agg_file_path)
        print(f"Saved Hacker News data for {symbol} -> {agg_file_path} ({len(df_raw)} stories)")

        time.sleep(1)  # be polite between assets


if __name__ == "__main__":
    main()
