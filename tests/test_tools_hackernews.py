import json
import time

import pytest

from radar.errors import ToolError
from radar.tools.hackernews import fetch_hackernews

HOUR = 3600


def hn_payload(hits):
    return json.dumps({"hits": hits})


def hn_hit(**overrides):
    hit = {
        "objectID": "42",
        "title": "A post about agents",
        "url": "https://example.com/agents",
        "author": "pg",
        "points": 120,
        "num_comments": 30,
        "created_at_i": int(time.time()) - HOUR,
        "story_text": "",
    }
    return {**hit, **overrides}


def fetcher(body, recorder=None):
    def fetch(url, **_kwargs):
        if recorder is not None:
            recorder.append(url)
        return body

    return fetch


def test_hackernews_normalises_hits_into_items():
    items = fetch_hackernews(fetch=fetcher(hn_payload([hn_hit()])))

    (item,) = items
    assert item.source == "hackernews"
    assert item.title == "A post about agents"
    assert item.url == "https://example.com/agents"
    assert item.author == "pg"
    assert item.score == 120
    assert item.comments == 30


def test_hackernews_asks_for_stories_within_the_window():
    urls: list[str] = []
    fetch_hackernews(hours=48, fetch=fetcher(hn_payload([]), urls))

    (url,) = urls
    assert "hn.algolia.com" in url
    assert "search_by_date" in url
    assert "created_at_i" in url, "the time window must be pushed to the server"


def test_hackernews_query_is_passed_through():
    urls: list[str] = []
    fetch_hackernews(query="prompt injection", fetch=fetcher(hn_payload([]), urls))

    assert "query=prompt+injection" in urls[0] or "query=prompt%20injection" in urls[0]


def test_hackernews_self_posts_fall_back_to_the_discussion_url():
    """Ask HN entries carry no external URL — without this they are unreachable."""
    items = fetch_hackernews(fetch=fetcher(hn_payload([hn_hit(url=None, objectID="999")])))

    assert items[0].url == "https://news.ycombinator.com/item?id=999"


def test_hackernews_drops_items_older_than_the_window():
    old = hn_hit(created_at_i=int(time.time()) - 100 * HOUR)
    fresh = hn_hit(objectID="7", created_at_i=int(time.time()) - 2 * HOUR)

    items = fetch_hackernews(hours=48, fetch=fetcher(hn_payload([old, fresh])))

    assert [i.url for i in items] == [fresh["url"]]


def test_hackernews_malformed_json_raises_tool_error():
    with pytest.raises(ToolError):
        fetch_hackernews(fetch=fetcher("<html>rate limited</html>"))


def test_hackernews_unexpected_shape_raises_tool_error():
    with pytest.raises(ToolError):
        fetch_hackernews(fetch=fetcher(json.dumps({"unexpected": True})))
