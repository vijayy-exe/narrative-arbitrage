"""Tests for src/data_pipeline/fourchan_biz.py. No live network calls --
mocked via requests_mock, following the same convention as
tests/data_pipeline/test_data.py.
"""
import pytest

from src.data_pipeline import fourchan_biz


def _catalog_page(threads):
    return [{"page": 1, "threads": threads}]


def test_search_catalog_matches_subject_and_comment_case_insensitively():
    threads = [
        {"no": 1, "sub": "GME to the moon", "com": ""},
        {"no": 2, "sub": "", "com": "just bought more gme, wagmi"},
        {"no": 3, "sub": "unrelated thread", "com": "nothing about the ticker here"},
    ]
    matches = fourchan_biz.search_catalog(threads, "gme")
    assert {t["no"] for t in matches} == {1, 2}


def test_strip_html_removes_tags_and_unescapes_entities():
    raw = "check this out&gt; <a href=\"#p123\" class=\"quotelink\">&gt;&gt;123</a><br>bullish af"
    text = fourchan_biz._strip_html(raw)
    assert "<" not in text and ">" not in text or "&gt;" not in text
    assert "bullish af" in text


def test_fetch_biz_snapshot_counts_posts_replies_and_sentiment(requests_mock):
    requests_mock.get(
        fourchan_biz.CATALOG_URL,
        json=_catalog_page([
            {"no": 100, "sub": "DOGE general", "com": "great news for doge today"},
            {"no": 200, "sub": "unrelated", "com": "nothing here"},
        ]),
    )
    requests_mock.get(
        fourchan_biz.THREAD_URL.format(thread_no=100),
        json={"posts": [
            {"no": 100, "com": "DOGE general"},  # OP, skipped (index 0)
            {"no": 101, "com": "this is terrible, awful, bad news actually"},
            {"no": 102, "com": "no it's great, amazing, wonderful"},
        ]},
    )

    result = fourchan_biz.fetch_biz_snapshot("doge", sleep_between_threads=0)

    assert result["post_count"] == 1  # only thread 100 matched "doge"
    assert result["comment_count"] == 2  # 2 replies in thread 100
    assert isinstance(result["sentiment"], float)


def test_fetch_biz_snapshot_returns_genuine_zero_when_nothing_matches(requests_mock):
    requests_mock.get(
        fourchan_biz.CATALOG_URL,
        json=_catalog_page([{"no": 1, "sub": "totally unrelated", "com": "nothing to see"}]),
    )

    result = fourchan_biz.fetch_biz_snapshot("nonexistent_ticker_xyz", sleep_between_threads=0)

    assert result == {"post_count": 0, "comment_count": 0, "sentiment": 0.0}


def test_fetch_thread_replies_returns_empty_list_on_404(requests_mock):
    requests_mock.get(fourchan_biz.THREAD_URL.format(thread_no=999), status_code=404)
    assert fourchan_biz.fetch_thread_replies(999) == []


def test_fetch_biz_snapshot_respects_max_threads_cap(requests_mock):
    threads = [{"no": i, "sub": "DOGE", "com": ""} for i in range(10)]
    requests_mock.get(fourchan_biz.CATALOG_URL, json=_catalog_page(threads))
    for t in threads:
        requests_mock.get(fourchan_biz.THREAD_URL.format(thread_no=t["no"]), json={"posts": [{"no": t["no"]}]})

    result = fourchan_biz.fetch_biz_snapshot("doge", sleep_between_threads=0, max_threads=3)

    assert result["post_count"] == 3
