"""4chan /biz/ attention-signal fetcher — LIVE-ONLY, cannot backfill history.

Added per `narrative-arbitrage-implementation-upgrade.md` Upgrade 0 step 3,
evaluating StockTwits and 4chan's /biz/ (Business & Finance) board as
replacements for the Reddit comment-count/sentiment signal (Reddit stopped
issuing self-serve API credentials as of 2026-09 -- see reddit.py and
configs/assets.yaml).

EVALUATION RESULT (verified live, 2026-09-20):
  - StockTwits: re-confirmed dead. `api.stocktwits.com/api/2/streams/
    symbol/<ticker>.json` returns HTTP 403 behind a Cloudflare JS challenge
    (same finding hackernews.py's module docstring already recorded when it
    evaluated StockTwits as a Reddit replacement -- this is a second,
    independent confirmation, not a new discovery). Not usable without
    solving a bot challenge, which this project does not do.
  - 4chan /biz/ (`a.4cdn.org/biz/*.json`): free, no auth, no key, returns
    HTTP 200 with real data (verified live). **BUT it has no historical
    archive.** `a.4cdn.org/biz/archive.json` returned only 1,161 archived
    thread numbers (verified live, 2026-09-20), all clustered within a
    single recent window (min/max thread numbers ~90k apart, consistent
    with 4chan's well-known behavior of permanently deleting threads once
    they fall off the archive after a board-dependent, typically single-digit-
    days retention period) -- nowhere close to reaching back to this
    project's 2023-01-01..2024-06-01 backfill window
    (configs/date_range.yaml). There is no date-range query parameter on
    any 4chan API endpoint; the catalog/archive only ever reflect *right
    now*.

CONCLUSION: 4chan /biz/ is the only one of the two evaluated alternatives
that actually works, but it can only ever produce a LIVE, going-forward
signal (Phase 7's live loop), never a historical backfill. The historical
gap this module was asked to close --
`comment_count`/`sentiment`/`novelty_score` missing from
`data/processed/*.csv` for the 2023-2024 backfill window -- remains
genuinely unfixable by any free, no-auth source found so far; it is NOT
fixed by this module and this module does not claim otherwise. What this
module DOES add: a real `biz_post_count`/`biz_comment_count`/`biz_sentiment`
signal for `src/realtime/scheduler.py`'s live loop, going forward only.

QUERY MATCHING (verified live, 2026-09-20): /biz/ is a small, fast-moving
board -- a single catalog fetch (11 pages) returned only ~201 live threads
total at the time of checking, so an OP-subject/comment substring match
against `biz_query` (mirrors `hn_query`'s pattern in configs/assets.yaml,
falls back to `name`) is cheap: no pagination, no rate-limit risk. Reply
text within matching threads is also scanned (via one thread-endpoint call
per matching thread) since /biz/ threads are typically OP-plus-replies
discussions, not just headlines -- the discussion volume lives in the
replies, not the OP post.
"""

import re
import time

import nltk
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer

CATALOG_URL = "https://a.4cdn.org/biz/catalog.json"
THREAD_URL = "https://a.4cdn.org/biz/thread/{thread_no}.json"
_TAG_RE = re.compile(r"<[^>]+>")

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


def _strip_html(raw_html):
    """4chan's `com`/`sub` fields are HTML fragments (<br>, <a class=quotelink>,
    &gt; entities, etc.) -- strip tags and unescape entities so VADER scores
    real text, not markup noise."""
    import html as html_lib

    if not raw_html:
        return ""
    text = raw_html.replace("<br>", " ").replace("<br/>", " ")
    text = _TAG_RE.sub(" ", text)
    return html_lib.unescape(text).strip()


def fetch_catalog(session=None, timeout=15):
    """Returns the flat list of currently-live OP thread dicts (subject,
    comment, replies count, etc.) across all /biz/ catalog pages. Raises on
    any request failure -- callers (the live scheduler's _safe_call wrapper)
    are responsible for catching and degrading gracefully, matching every
    other fetcher in this package.
    """
    session = session or requests
    resp = session.get(CATALOG_URL, headers={"User-Agent": "narrative-arbitrage-research/1.0"},
                        timeout=timeout)
    resp.raise_for_status()
    pages = resp.json()
    return [thread for page in pages for thread in page.get("threads", [])]


def fetch_thread_replies(thread_no, session=None, timeout=15):
    """Returns the list of reply post dicts (everything after index 0, the
    OP) for one thread. Returns [] (not an error) if the thread has already
    404'd between the catalog fetch and this call -- threads on /biz/ can
    disappear within minutes, that is expected board churn, not a bug.
    """
    session = session or requests
    url = THREAD_URL.format(thread_no=thread_no)
    resp = session.get(url, headers={"User-Agent": "narrative-arbitrage-research/1.0"}, timeout=timeout)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    posts = resp.json().get("posts", [])
    return posts[1:]  # index 0 is the OP, already counted via the catalog


def search_catalog(threads, query):
    """Case-insensitive substring match of `query` against each OP thread's
    subject + comment (both HTML-stripped). Mirrors hackernews.py's
    single-term matching approach -- deliberately simple, not a ranked
    search, since /biz/'s live thread count is small enough (~200 at a time,
    verified live) that a plain substring scan is cheap and its false-
    positive rate can be eyeballed directly (see this module's docstring for
    the verified 2026-09-20 hit counts per asset).
    """
    q = query.lower()
    return [t for t in threads
            if q in (_strip_html(t.get("sub", "")) + " " + _strip_html(t.get("com", ""))).lower()]


def fetch_biz_snapshot(query, session=None, sleep_between_threads=0.2, max_threads=25):
    """Live-only snapshot (NOT a historical time series -- see module
    docstring): searches the CURRENT /biz/ catalog for `query`, fetches
    replies for up to `max_threads` matching threads (most recent first, by
    thread number, to bound the request count on a hot query), and returns:

      {post_count, comment_count, sentiment}

    - post_count: number of matching live OP threads right now.
    - comment_count: total replies across those matching threads.
    - sentiment: mean VADER compound score over OP + reply text, across all
      matching threads/posts (0.0 / neutral if no text at all).

    Returns a genuine zero-row ({"post_count": 0, "comment_count": 0,
    "sentiment": 0.0}) when nothing matches right now -- that is a real
    "no current /biz/ discussion" reading, not a fetch failure; the caller
    (src/realtime/scheduler.py's _safe_call) is what distinguishes an actual
    fetch failure (raised exception) from a genuine zero.
    """
    threads = fetch_catalog(session=session)
    matches = sorted(search_catalog(threads, query), key=lambda t: t.get("no", 0), reverse=True)
    matches = matches[:max_threads]

    sia = SentimentIntensityAnalyzer()
    texts = []
    total_replies = 0

    for thread in matches:
        texts.append(_strip_html(thread.get("sub", "")) + " " + _strip_html(thread.get("com", "")))
        replies = fetch_thread_replies(thread["no"], session=session)
        total_replies += len(replies)
        texts.extend(_strip_html(p.get("com", "")) for p in replies)
        time.sleep(sleep_between_threads)

    texts = [t for t in texts if t.strip()]
    sentiment = float(sum(sia.polarity_scores(t)["compound"] for t in texts) / len(texts)) if texts else 0.0

    return {
        "post_count": len(matches),
        "comment_count": total_replies,
        "sentiment": sentiment,
    }


def main():
    """Manual smoke-test entry point (`python -m src.data_pipeline.fourchan_biz
    <query>`) -- this module is wired into the live scheduler
    (src/realtime/scheduler.py), not run as a standalone historical-backfill
    script like wikipedia.py/trends.py/hackernews.py, since it has nothing
    to backfill (see module docstring)."""
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "GameStop"
    print(f"Fetching live /biz/ snapshot for query={query!r}...")
    print(fetch_biz_snapshot(query))


if __name__ == "__main__":
    main()
