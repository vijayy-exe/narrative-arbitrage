# Feature Dictionary

Documents every column that `src/data_pipeline/features.py` (`process_asset`)
can write to `data/processed/<asset_symbol>_features.csv`. Not every column
appears in every file: Reddit-derived columns only appear when
`data/raw/reddit/<symbol>.csv` (and `_raw.csv`) exist, which currently
requires live Reddit API credentials (see `.env.example` and
`src/data_pipeline/reddit.py`) that are not obtainable self-serve as of
2026-09 (Reddit now gates script-app credentials behind a manual review --
see `configs/assets.yaml`). Hacker News (`hn_*` columns, via
`src/data_pipeline/hackernews.py`) is the real, no-auth forum-attention
signal used in its place; it is present for every asset in this run.
GDELT (news volume/tone) and StockTwits (financial-community sentiment) were
evaluated as additional sources and dropped -- see
`src/data_pipeline/hackernews.py`'s module docstring for why (GDELT: 429
rate-limited on every attempt, including from a separate fetch path;
StockTwits: anonymous endpoint now behind a Cloudflare bot challenge --
re-confirmed live 2026-09-20, see `src/data_pipeline/fourchan_biz.py`'s
module docstring for the second, independent confirmation). 4chan's /biz/
board was also evaluated (2026-09-20) as a Reddit comment/sentiment
replacement and, unlike StockTwits/GDELT, genuinely works (free, no auth) --
but it has no historical archive (verified live: only ~1,161 archived
threads exist at any time, all recent, nowhere close to reaching back to
this project's 2023-2024 backfill window), so it can only ever produce a
LIVE, going-forward signal via `src/realtime/scheduler.py`
(`live_biz_signals` table), never a `data/processed/*.csv` backfill column.
**`comment_count`/`sentiment`/`novelty_score` remain absent from every
historical processed CSV and cannot be filled retroactively by any free,
no-auth source found so far** -- this is a permanent gap for the 2023-2024
window, not a pending fix.
Columns are listed in the order they're computed, not the order they appear
in the CSV.

All values are causal: every column at row `t` is computed using only rows
`<= t` from the raw inputs. This is enforced by
`tests/data_pipeline/test_no_leakage.py`, not just documented here (see that
test's docstring for the exact method: compute twice, once truncated and
once with future rows appended, and assert identical output for the
overlapping timestamps).

## Raw / passthrough columns

| Column | Source | Type | Meaning |
|---|---|---|---|
| `date` | all | date (index) | Calendar day (UTC), the join key across all sources. |
| `pageviews` | Wikipedia (Wikimedia Pageviews REST API) | int | Total daily pageviews (all-access, all-agents) for the asset's Wikipedia article (`wikipedia_title` in `configs/assets.yaml`). |
| `edit_count` | Wikipedia (MediaWiki API) | int | Number of revisions made to the article that day. |
| `unique_editors` | Wikipedia (MediaWiki API) | int | Number of distinct usernames that made at least one edit that day. |
| `revert_count` | Wikipedia (MediaWiki API) | int | Number of that day's edits whose comment matched a revert keyword (`revert`, `undid`, `rolled back`, `undo`) -- a heuristic, not the MediaWiki "revert" flag, so it can under/over-count relative to a stricter definition. |
| `post_count` | Reddit (PRAW search across `subreddits`) | int | Number of matching submissions (posts) that day, summed across the configured subreddit list. Subject to the ~1000-result search cap described in `reddit.py` -- NOT guaranteed to be a complete daily count over a long backfill window. |
| `comment_count` | Reddit | int | Sum of `num_comments` (Reddit's own comment-count metadata) across matching submissions that day. Same completeness caveat as `post_count`. |
| `sentiment` | Reddit (VADER) | float, [-1, 1] | Mean VADER compound sentiment score of (title + selftext) across that day's matching submissions. |
| `search_interest` | Google Trends (pytrends) | int, [0, 100] | Google's normalized search-interest index for the asset's `name`. **Reported weekly, not daily** (a pytrends/Google Trends characteristic for date ranges longer than ~9 months) -- see Missing-Data Handling below for how this is reconciled with the daily bar grid. |
| `hn_story_count` | Hacker News (Algolia search API) | int | Number of HN stories that day whose title/text matched the asset's `hn_query` (falls back to `name` -- see `configs/assets.yaml`). |
| `hn_points` | Hacker News | int | Sum of those stories' point (upvote) totals for that day. |
| `hn_comment_count` | Hacker News | int | Sum of those stories' comment counts for that day. |

## Derived: first/second derivatives (velocity / acceleration)

Computed by `calculate_derivatives()` for each of `pageviews`, `edit_count`,
`comment_count`, `search_interest` (whichever are present):

| Column pattern | Formula | Meaning |
|---|---|---|
| `<signal>_velocity` | `signal.diff()` i.e. `signal[t] - signal[t-1]` | First derivative: day-over-day change. `NaN` on the first row (no prior day). |
| `<signal>_acceleration` | `velocity.diff()` i.e. `velocity[t] - velocity[t-1]` | Second derivative: change in the rate of change. `NaN` on the first two rows. |

So in practice: `pageviews_velocity`, `pageviews_acceleration`,
`edit_count_velocity`, `edit_count_acceleration`, `comment_count_velocity`,
`comment_count_acceleration`, `search_interest_velocity`,
`search_interest_acceleration`, `hn_story_count_velocity`,
`hn_story_count_acceleration`, `hn_points_velocity`,
`hn_points_acceleration`, `hn_comment_count_velocity`,
`hn_comment_count_acceleration`.

## Derived: other attention features

| Column | Formula | Meaning |
|---|---|---|
| `edit_war_ratio` | `revert_count / (edit_count + 1)` | Fraction of that day's edits that look like reverts (the `+1` avoids divide-by-zero on days with no edits, at the cost of slightly deflating the ratio on low-edit-count days -- this is an intentional, documented tradeoff, not a bug). Range roughly `[0, 1]`. |
| `novelty_score` | `1 - cosine_similarity(TF-IDF(today's Reddit text), mean(TF-IDF(trailing 7 days of Reddit text)))` | How much today's Reddit discussion content differs from the trailing 7-day window (`calculate_novelty_score()`, `window=7`). `0` = today's text looks just like the recent past; closer to `1` = novel/different vocabulary. `NaN` for the first `window` rows (not enough history), and `NaN` on any day where the TF-IDF vocabulary is empty (e.g. no Reddit text at all, or only stop words, in the current+past window) -- this is the expected/benign `ValueError` case, not a bug (see the narrowed exception handling in `calculate_novelty_score`). Requires `data/raw/reddit/<symbol>_raw.csv` (the raw per-post text), not just the aggregated `<symbol>.csv`. |
| `sentiment_divergence` | `sentiment - normalized(edit_count_velocity)` | Difference between Reddit sentiment and Wikipedia edit *velocity* (normalized to roughly `[0, 1]`), as a simple proxy for cross-platform divergence. Normalization uses an **expanding (causal) min/max** -- i.e. at row `t` it only uses `edit_count_velocity` values from rows `0..t`, never later rows. (This was fixed during the Phase 1 review: the original code normalized with the min/max over the *entire* series, which leaked future values into every historical row's normalization -- see the LEAKAGE FIX comment in `features.py`. The causal version means early rows are normalized against very little history and the effective normalization range can shift over time; this is an accepted, documented tradeoff for being leak-free, not an oversight.) Only computed when both `sentiment` and `edit_count_velocity` are present, i.e. only when Reddit data exists. |

## Missing-data handling (per source)

Decisions made in `process_asset()`, and why -- this affects leakage risk,
per the dev plan's explicit ask to document the choice:

- **Wikipedia / Reddit** (`pageviews`, `edit_count`, `unique_editors`,
  `revert_count`, `post_count`, `comment_count`, `sentiment`): missing rows
  after the outer join are filled with `0`. A day absent from the raw fetch
  genuinely means "nothing happened" (no edits, no matching posts) or "not
  fetched" -- either way, `0` is the right causal fill; it never uses a
  future value.
- **Google Trends** (`search_interest`): filled with **forward-fill**
  (`.ffill()`), not zero. Trends data is reported weekly; zero-filling the
  six days between reports would fabricate a sawtooth pattern (a real value
  on the report day, a fake collapse to 0 for six days, then a fake
  vertical velocity/acceleration spike every week) that is a bigger
  distortion than the missingness itself. Forward-filling propagates the
  last *known, past* reading forward, which is still strictly causal
  (it never reads a future value) and produces a much more sensible
  step-function `search_interest_velocity`/`_acceleration` (mostly flat,
  with the real jump occurring the week interest was actually reported).
- **Missing entire source** (most commonly Reddit, since it requires live
  API credentials that may not be configured): `process_asset()` proceeds
  with whichever of {Wikipedia, Reddit, Trends} are actually present,
  requiring at least one of Wikipedia or Trends. It prints which sources
  were unavailable for each asset. Reddit-derived columns
  (`comment_count*`, `sentiment`, `novelty_score`, `sentiment_divergence`)
  are simply absent from the CSV in that case, rather than being
  fabricated or zero-filled as if Reddit had been checked and found quiet.

## Known caveats affecting interpretation

- **PEPE's Wikipedia signal is a proxy, not asset-specific.** There is no
  dedicated Wikipedia article for the Pepe *cryptocurrency* (verified via
  the MediaWiki API: `Pepe (cryptocurrency)`, `Pepecoin`, and
  `Pepe (memecoin)` are all missing pages). `configs/assets.yaml` uses
  `Pepe the Frog` (the meme) as the closest available real page, so PEPE's
  `pageviews`/`edit_count`/etc. reflect general meme attention, not
  coin-specific attention.
- **Reddit's ~1000-result search cap** (see `src/data_pipeline/reddit.py`
  module docstring) means `post_count`/`comment_count`/`sentiment` are not
  guaranteed to be a complete daily series over a long (~17 month) backfill
  window, even once credentials are configured. Treat gaps as expected, not
  as "zero attention that day."
- **`hn_*` signals are real but sparse for most assets.** Total HN volume
  for a niche/mid-cap meme-stock or small-cap-crypto term over the 17-month
  window is small (verified live: 2 stories for SHIB/CryptoPunks, 91 for
  Bored Ape -- see `configs/assets.yaml`'s `hn_query` comment block), so
  most days genuinely have `hn_story_count == 0`. That is the correct value,
  not a fetch gap -- unlike Reddit's cap, this is a real ceiling on how much
  HN discussion these assets get, not an API limitation.
