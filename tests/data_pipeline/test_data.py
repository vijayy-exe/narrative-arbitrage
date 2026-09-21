"""Real, mocked-network tests for each raw-data fetcher in src/data_pipeline/.

None of these tests hit a live API -- HTTP calls are mocked with
`requests_mock`, and the PRAW/pytrends/yfinance/ccxt clients are mocked at
the Python-object level, all over a small (2-3 day) date range. They assert
the shape/columns of the resulting DataFrame, not just "it didn't crash".
"""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_pipeline import wikipedia, reddit, trends, price


# ---------------------------------------------------------------------------
# wikipedia.py
# ---------------------------------------------------------------------------

def test_fetch_pageviews_mocked(requests_mock):
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        "en.wikipedia.org/all-access/all-agents/GameStop/daily/20230101/20230103"
    )
    requests_mock.get(url, json={
        "items": [
            {"timestamp": "2023010100", "views": 1000},
            {"timestamp": "2023010200", "views": 1500},
            {"timestamp": "2023010300", "views": 900},
        ]
    })

    df = wikipedia.fetch_pageviews("GameStop", "2023-01-01", "2023-01-03")

    assert list(df.columns) == ["pageviews"]
    assert len(df) == 3
    assert df.loc[pd.to_datetime("2023-01-02").date(), "pageviews"] == 1500


def test_fetch_edits_paginates_using_continue_token(requests_mock):
    """Regression test for the pagination bug: the fixed implementation must
    (a) use the server-provided `continue`/`rvcontinue` token to advance, and
    (b) stop once the LATEST (not earliest) revision in a batch passes
    end_date, without re-fetching the same window forever."""
    url = "https://en.wikipedia.org/w/api.php"

    first_page = {
        "continue": {"rvcontinue": "20230102000000|2", "continue": "||"},
        "query": {
            "pages": {
                "123": {
                    "pageid": 123,
                    "revisions": [
                        {"user": "alice", "timestamp": "2023-01-01T10:00:00Z", "comment": "initial edit"},
                        {"user": "bob", "timestamp": "2023-01-02T10:00:00Z", "comment": "reverted vandalism"},
                    ]
                }
            }
        }
    }
    second_page = {
        # No "continue" key -> this is the last page.
        "query": {
            "pages": {
                "123": {
                    "pageid": 123,
                    "revisions": [
                        {"user": "carol", "timestamp": "2023-01-03T10:00:00Z", "comment": "typo fix"},
                    ]
                }
            }
        }
    }

    requests_mock.get(url, [{"json": first_page}, {"json": second_page}])

    df = wikipedia.fetch_edits("GameStop", "2023-01-01", "2023-01-03")

    # Exactly 2 requests -- one per page, no runaway re-fetching of the same window.
    assert requests_mock.call_count == 2
    # The second request must have used the server's continuation token.
    second_request_qs = requests_mock.request_history[1].qs
    assert second_request_qs.get("rvcontinue") == ["20230102000000|2"]

    assert len(df) == 3
    assert df.loc[pd.to_datetime("2023-01-01").date(), "edit_count"] == 1
    assert df.loc[pd.to_datetime("2023-01-02").date(), "revert_count"] == 1  # "reverted vandalism"
    assert df.loc[pd.to_datetime("2023-01-03").date(), "revert_count"] == 0


def test_fetch_edits_stops_without_looping_on_missing_page(requests_mock):
    url = "https://en.wikipedia.org/w/api.php"
    requests_mock.get(url, json={"query": {"pages": {"-1": {"missing": ""}}}})

    df = wikipedia.fetch_edits("Not A Real Page At All", "2023-01-01", "2023-01-03")

    assert requests_mock.call_count == 1  # must not loop forever on a missing page
    assert df.empty


# ---------------------------------------------------------------------------
# reddit.py
# ---------------------------------------------------------------------------

def _fake_submission(title, selftext, created_utc, num_comments):
    sub = MagicMock()
    sub.title = title
    sub.selftext = selftext
    sub.created_utc = created_utc
    sub.num_comments = num_comments
    return sub


def test_fetch_reddit_data_filters_by_date_and_scores_sentiment():
    in_range_ts = pd.Timestamp("2023-01-02", tz="UTC").timestamp()
    out_of_range_ts = pd.Timestamp("2022-06-01", tz="UTC").timestamp()

    submissions = [
        _fake_submission("This is amazing, great news for everyone", "excellent rally, love it", in_range_ts, 42),
        _fake_submission("old post, out of range", "", out_of_range_ts, 5),
    ]

    fake_reddit = MagicMock()
    fake_reddit.subreddit.return_value.search.return_value = submissions

    df = reddit.fetch_reddit_data(
        fake_reddit, "GameStop", ["Superstonk", "GME"], "2023-01-01", "2023-01-03"
    )

    fake_reddit.subreddit.assert_called_once_with("Superstonk+GME")
    assert len(df) == 1  # the out-of-range submission must be filtered out
    row = df.iloc[0]
    assert row["comment_count"] == 42
    assert row["sentiment"] > 0  # "moon"/"diamond hands" should score positive with VADER


def test_get_reddit_client_returns_none_without_credentials(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)

    assert reddit.get_reddit_client() is None


def test_get_reddit_client_sets_read_only(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "test-agent")

    with patch("src.data_pipeline.reddit.praw.Reddit") as mock_reddit_cls:
        mock_instance = MagicMock()
        mock_reddit_cls.return_value = mock_instance

        client = reddit.get_reddit_client()

        assert client is mock_instance
        assert mock_instance.read_only is True


# ---------------------------------------------------------------------------
# trends.py
# ---------------------------------------------------------------------------

def test_fetch_trends_mocked():
    fake_df = pd.DataFrame(
        {"GameStop": [10, 20, 30], "isPartial": [False, False, True]},
        index=pd.to_datetime(["2023-01-01", "2023-01-08", "2023-01-15"]),
    )

    with patch("src.data_pipeline.trends.TrendReq") as mock_trendreq_cls:
        mock_instance = MagicMock()
        mock_instance.interest_over_time.return_value = fake_df
        mock_trendreq_cls.return_value = mock_instance

        df = trends.fetch_trends("GameStop", "2023-01-01", "2023-01-15")

    assert list(df.columns) == ["search_interest"]
    assert len(df) == 3
    assert df["search_interest"].tolist() == [10, 20, 30]


def test_fetch_trends_empty_result_handled():
    with patch("src.data_pipeline.trends.TrendReq") as mock_trendreq_cls:
        mock_instance = MagicMock()
        mock_instance.interest_over_time.return_value = pd.DataFrame()
        mock_trendreq_cls.return_value = mock_instance

        df = trends.fetch_trends("SomeObscureTerm", "2023-01-01", "2023-01-15")

    assert df.empty


# ---------------------------------------------------------------------------
# price.py
# ---------------------------------------------------------------------------

def test_fetch_stock_price_flattens_multiindex_columns():
    dates = pd.to_datetime(["2023-01-03", "2023-01-04"])
    multi_cols = pd.MultiIndex.from_product([["Close", "Open", "High", "Low", "Volume"], ["GME"]])
    fake_data = pd.DataFrame(
        [[17.2, 18.6, 19.2, 17.0, 5135200], [17.3, 17.2, 17.9, 16.9, 3939300]],
        index=dates, columns=multi_cols,
    )

    with patch("src.data_pipeline.price.yf.download", return_value=fake_data):
        df = price.fetch_stock_price("GME", "2023-01-03", "2023-01-05")

    assert not isinstance(df.columns, pd.MultiIndex)
    assert list(df.columns) == ["Close", "Open", "High", "Low", "Volume"]
    assert df.index.name == "date"
    assert len(df) == 2


def test_fetch_crypto_price_trims_to_requested_range():
    # Simulate ccxt returning one batch whose last candle lands AFTER
    # end_date -- the fetcher must trim the output to [start, end], not just
    # stop looping (this was the actual bug: the out-of-range batch used to
    # get appended wholesale before the loop broke).
    day_ms = 86400000
    base = pd.Timestamp("2023-01-01", tz="UTC").value // 1_000_000
    ohlcv_batch = [
        [base + i * day_ms, 1.0, 1.1, 0.9, 1.0, 1000.0]
        for i in range(10)  # 2023-01-01 .. 2023-01-10
    ]

    fake_exchange = MagicMock()
    fake_exchange.parse8601 = lambda s: int(pd.Timestamp(s).value // 1_000_000)
    fake_exchange.fetch_ohlcv.side_effect = [ohlcv_batch, []]

    with patch("src.data_pipeline.price.ccxt.binance", return_value=fake_exchange):
        df = price.fetch_crypto_price("DOGE", "2023-01-01", "2023-01-05")

    assert df.index.max() <= pd.to_datetime("2023-01-05").date()
    assert df.index.min() >= pd.to_datetime("2023-01-01").date()
    assert len(df) == 5
