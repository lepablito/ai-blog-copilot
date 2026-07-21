import json
import time
from datetime import UTC, datetime

import pytest

from radar.errors import ToolError
from radar.tools.hackernews import fetch_hackernews
from radar.tools.reddit import fetch_reddit

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


def reddit_payload(posts):
    return json.dumps({"data": {"children": [{"kind": "t3", "data": p} for p in posts]}})


def reddit_post(**overrides):
    post = {
        "title": "Local models got good",
        "url": "https://example.com/local",
        "permalink": "/r/LocalLLaMA/comments/abc/local/",
        "author": "someone",
        "score": 300,
        "num_comments": 45,
        "created_utc": time.time() - HOUR,
        "selftext": "body text",
        "stickied": False,
    }
    return {**post, **overrides}


def fetcher(body, recorder=None):
    def fetch(url, **_kwargs):
        if recorder is not None:
            recorder.append(url)
        return body

    return fetch


# --- Hacker News ---


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


# --- Reddit ---


# The rich JSON listing is only reachable with a token, so every test that
# exercises it passes one. TOKEN's value is irrelevant — the fake fetcher never
# checks it; test_reddit_auth.py covers the header itself.
TOKEN = "test-token"


def test_reddit_normalises_posts_into_items():
    items = fetch_reddit(
        ["LocalLLaMA"], token=TOKEN, fetch=fetcher(reddit_payload([reddit_post()]))
    )

    (item,) = items
    assert item.source == "reddit"
    assert item.title == "Local models got good"
    assert item.score == 300
    assert item.comments == 45
    assert "body text" in item.text_excerpt


def test_reddit_skips_stickied_announcements():
    posts = [reddit_post(stickied=True), reddit_post(title="real post")]

    items = fetch_reddit(["LocalLLaMA"], token=TOKEN, fetch=fetcher(reddit_payload(posts)))

    assert [i.title for i in items] == ["real post"]


def test_reddit_drops_posts_older_than_the_window():
    posts = [reddit_post(created_utc=time.time() - 100 * HOUR), reddit_post(title="fresh")]

    items = fetch_reddit(
        ["LocalLLaMA"], hours=48, token=TOKEN, fetch=fetcher(reddit_payload(posts))
    )

    assert [i.title for i in items] == ["fresh"]


def test_reddit_queries_every_subreddit():
    urls: list[str] = []
    fetch_reddit(
        ["MachineLearning", "LocalLLaMA"], token=TOKEN, fetch=fetcher(reddit_payload([]), urls)
    )

    assert any("MachineLearning" in u for u in urls)
    assert any("LocalLLaMA" in u for u in urls)


def test_one_failing_subreddit_does_not_lose_the_others():
    """Reddit 403s aggressively. Losing one subreddit must not cost the rest."""

    def fetch(url, **_kwargs):
        if "Banned" in url:
            raise ToolError("HTTP 403")
        return reddit_payload([reddit_post(title="survived")])

    items = fetch_reddit(["Banned", "LocalLLaMA"], token=TOKEN, fetch=fetch)

    assert [i.title for i in items] == ["survived"]


def reddit_rss(entries):
    body = "".join(
        f"""<entry>
          <title>{e['title']}</title>
          <link href="{e['link']}"/>
          <updated>{e['when']}</updated>
          <author><name>/u/{e.get('author', 'someone')}</name></author>
        </entry>"""
        for e in entries
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"><title>r/x</title>{body}</feed>"""


def test_unauthenticated_run_reads_the_public_rss_listing():
    recent = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    calls: list[str] = []

    def fetch(url, **_kwargs):
        calls.append(url)
        return reddit_rss([{"title": "via rss", "link": "https://redd.it/1", "when": recent}])

    items = fetch_reddit(["LocalLLaMA"], fetch=fetch)

    assert [i.title for i in items] == ["via rss"]
    assert any("old.reddit.com" in c and ".rss" in c for c in calls)


def test_rss_items_carry_no_invented_score():
    recent = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    def fetch(_url, **_kwargs):
        return reddit_rss([{"title": "via rss", "link": "https://redd.it/1", "when": recent}])

    (item,) = fetch_reddit(["LocalLLaMA"], fetch=fetch)

    assert item.source == "reddit"
    assert item.score == 0, "the rss listing carries no score — do not fabricate one"


def test_both_reddit_paths_failing_is_a_failure_for_that_subreddit():
    def fetch(_url, **_kwargs):
        raise ToolError("HTTP 403")

    with pytest.raises(ToolError, match="403"):
        fetch_reddit(["LocalLLaMA"], fetch=fetch)


def test_every_subreddit_failing_raises_tool_error():
    def fetch(_url, **_kwargs):
        raise ToolError("HTTP 403")

    with pytest.raises(ToolError, match="403"):
        fetch_reddit(["a", "b"], fetch=fetch)


def test_reddit_post_without_external_url_uses_its_permalink():
    post = reddit_post(url="", permalink="/r/x/comments/abc/title/")

    items = fetch_reddit(["x"], token=TOKEN, fetch=fetcher(reddit_payload([post])))

    assert items[0].url == "https://www.reddit.com/r/x/comments/abc/title/"
