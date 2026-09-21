import time
import yaml
import requests
import pandas as pd
from pathlib import Path


def load_config():
    with open("configs/assets.yaml", "r") as f:
        assets = yaml.safe_load(f)
    with open("configs/date_range.yaml", "r") as f:
        dates = yaml.safe_load(f)
    return assets, dates


def iter_assets(assets):
    """Flatten configs/assets.yaml into (symbol, wikipedia_title) pairs."""
    for stock in assets.get("meme_stocks", []):
        yield stock["ticker"], stock["wikipedia_title"]
    for crypto in assets.get("small_cap_crypto", []):
        yield crypto["symbol"], crypto["wikipedia_title"]
    for nft in assets.get("nft_collections", []):
        yield nft["id"], nft["wikipedia_title"]


def fetch_pageviews(title, start_date, end_date):
    print(f"Fetching Wikipedia pageviews for {title}...")
    # Formatting dates to YYYYMMDD
    s = start_date.replace("-", "")
    e = end_date.replace("-", "")
    # Pageviews REST API wants underscores, not spaces, in the article path.
    article = title.replace(" ", "_")
    # Correct URL structure: project/access/agent/article/granularity/start/end
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/{article}/daily/{s}/{e}"

    headers = {"User-Agent": "NarrativeArbitrageBot/1.0 (research project; contact: emveehehee@gmail.com)"}

    # Basic retry logic for 429s
    for attempt in range(3):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            break
        elif response.status_code == 429:
            wait = (attempt + 1) * 5
            print(f"Rate limited (429). Waiting {wait}s...")
            time.sleep(wait)
        else:
            print(f"Failed to fetch pageviews for {title}: {response.status_code}")
            return pd.DataFrame()
    else:
        print(f"Max retries exceeded for {title} pageviews.")
        return pd.DataFrame()

    data = response.json()
    if 'items' not in data:
        print(f"No items found in pageview response for {title}")
        return pd.DataFrame()

    df = pd.DataFrame(data['items'])
    df['date'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H').dt.date
    df = df.set_index('date')[['views']]
    df.columns = ['pageviews']
    return df


def fetch_edits(title, start_date, end_date):
    """Fetch daily edit/editor/revert counts for `title` between start_date
    and end_date (inclusive) via the MediaWiki API, paginating with the
    server-provided `rvcontinue` continuation token.

    Fix history: the original implementation used `rvdir="newer"` (so each
    fetched batch of revisions is in ASCENDING timestamp order) but advanced
    the loop and checked the end-date cutoff using `revisions[0]['timestamp']`
    -- the EARLIEST revision in the batch, not the latest. That meant:
      1. `rvstart` was barely advanced each iteration (re-fetching almost the
         same window repeatedly -- could loop for a very long time on a
         heavily-edited page and hammers the API pointlessly), and
      2. the end-date check could fire prematurely, since the earliest
         revision in a batch reaches `end_date` long before the latest one
         does, silently truncating history.
    The fix uses `revisions[-1]['timestamp']` (the newest revision in the
    batch) for both continuation and the end-date check, and prefers the
    API's own `continue`/`rvcontinue` token over manually recomputed
    `rvstart` whenever the server supplies one (this is the documented,
    robust way to paginate the MediaWiki API).
    """
    print(f"Fetching Wikipedia edit history for {title}...")
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "NarrativeArbitrageBot/1.0 (research project; contact: emveehehee@gmail.com)"}
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": "500",
        "rvstart": f"{start_date}T00:00:00Z",
        "rvend": f"{end_date}T23:59:59Z",
        "rvdir": "newer",
        "rvprop": "user|timestamp|comment",
        "format": "json",
    }

    all_revisions = []
    max_pages = 200  # safety valve: 200 * 500 = 100k revisions, far more than any of our pages need

    for _ in range(max_pages):
        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch edits: {response.status_code}")
            break

        try:
            res = response.json()
        except requests.exceptions.JSONDecodeError:
            print("Failed to decode JSON from edit history response.")
            break

        pages = res.get("query", {}).get("pages", {})
        if not pages:
            break

        page_id = list(pages.keys())[0]
        if page_id == "-1":
            print(f"Wikipedia page not found: {title!r}")
            break

        revisions = pages[page_id].get("revisions", [])
        if not revisions:
            break

        all_revisions.extend(revisions)

        # The latest revision in this batch (rvdir="newer" -> ascending order,
        # so it's the LAST element, not the first).
        latest_rev_timestamp = revisions[-1]['timestamp']

        continue_block = res.get("continue", {})
        if not continue_block:
            # No continuation token -> we've reached the end of history.
            break
        if latest_rev_timestamp > f"{end_date}T23:59:59Z":
            # We've already fetched past the end of our window.
            break

        # Prefer the server's own continuation token over manually bumping
        # rvstart -- it's the documented, robust way to paginate.
        params.update(continue_block)
        time.sleep(0.2)  # be polite to the API
    else:
        print(f"Hit max_pages={max_pages} safety limit for {title}; edit history may be incomplete.")

    df_revs = pd.DataFrame(all_revisions)
    if df_revs.empty:
        return pd.DataFrame()

    df_revs['timestamp'] = pd.to_datetime(df_revs['timestamp']).dt.date

    # Calculate metrics per day
    daily_metrics = df_revs.groupby('timestamp').agg(
        edit_count=('user', 'count'),
        unique_editors=('user', 'nunique')
    )

    # Revert count: looking for keywords like "reverted", "undid", "rolled back" in comments
    def is_revert(comment):
        if not isinstance(comment, str) or not comment:
            return 0
        comment = comment.lower()
        revert_keywords = ["revert", "undid", "rolled back", "undo"]
        return 1 if any(k in comment for k in revert_keywords) else 0

    df_revs['is_revert'] = df_revs['comment'].apply(is_revert)
    revert_counts = df_revs.groupby('timestamp')['is_revert'].sum()

    daily_metrics['revert_count'] = revert_counts
    daily_metrics.index.name = 'date'
    return daily_metrics


def main():
    assets, dates = load_config()
    start = dates['start_date']
    end = dates['end_date']

    raw_dir = Path("data/raw/wikipedia")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for symbol, title in iter_assets(assets):
        file_path = raw_dir / f"{symbol}.csv"
        if file_path.exists():
            print(f"Wikipedia data for {symbol} already exists. Skipping.")
            continue

        views = fetch_pageviews(title, start, end)
        edits = fetch_edits(title, start, end)

        if views.empty and edits.empty:
            print(f"Missing data for {symbol} ({title})")
            continue

        df = views.join(edits, how='outer').fillna(0)
        df.to_csv(file_path)
        print(f"Saved Wikipedia data for {symbol} -> {file_path}")


if __name__ == "__main__":
    main()
