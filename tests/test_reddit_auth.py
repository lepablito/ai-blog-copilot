import base64
from datetime import UTC, datetime

import httpx
import pytest
import respx

from radar.errors import ToolError
from radar.tools.reddit import OAUTH_HOST, fetch_reddit, get_app_token

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"


# --- getting a token ---


@respx.mock
def test_token_request_uses_basic_auth_and_client_credentials():
    route = respx.post(TOKEN_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "tok-123", "expires_in": 86400})
    )

    assert get_app_token("my-id", "my-secret") == "tok-123"

    request = route.calls.last.request
    expected = base64.b64encode(b"my-id:my-secret").decode()
    assert request.headers["authorization"] == f"Basic {expected}"
    assert b"grant_type=client_credentials" in request.read()


@respx.mock
def test_token_request_identifies_itself():
    """Reddit rejects anonymous clients — the UA is part of the contract."""
    route = respx.post(TOKEN_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t", "expires_in": 1})
    )

    get_app_token("id", "secret")

    assert "ai-blog-copilot" in route.calls.last.request.headers["user-agent"]


@respx.mock
def test_bad_credentials_raise_tool_error():
    respx.post(TOKEN_URL).mock(return_value=httpx.Response(401, json={"error": "invalid_client"}))

    with pytest.raises(ToolError):
        get_app_token("id", "wrong")


@respx.mock
def test_token_response_without_a_token_raises_tool_error():
    respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={"unexpected": True}))

    with pytest.raises(ToolError):
        get_app_token("id", "secret")


# --- using the token ---


def listing(titles):
    import json

    now = datetime.now(UTC).timestamp()
    return json.dumps(
        {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": t,
                            "url": f"https://example.com/{t}",
                            "permalink": f"/r/x/{t}",
                            "author": "someone",
                            "score": 1234,
                            "num_comments": 56,
                            "created_utc": now - 3600,
                            "selftext": "",
                            "stickied": False,
                        }
                    }
                    for t in titles
                ]
            }
        }
    )


def test_authenticated_run_hits_the_oauth_host_with_a_bearer_token():
    seen: list[tuple[str, dict]] = []

    def fetch(url, **kwargs):
        seen.append((url, kwargs.get("headers") or {}))
        return listing(["authenticated post"])

    items = fetch_reddit(["LocalLLaMA"], token="tok-123", fetch=fetch)

    url, headers = seen[0]
    assert OAUTH_HOST in url
    assert headers["Authorization"] == "Bearer tok-123"
    assert items[0].score == 1234, "the whole point of authenticating is getting the score back"


def test_one_token_serves_every_subreddit():
    calls: list[str] = []

    def fetch(url, **_kwargs):
        calls.append(url)
        return listing(["p"])

    fetch_reddit(["a", "b", "c"], token="tok", fetch=fetch)

    assert len(calls) == 3
    assert all(OAUTH_HOST in c for c in calls)


def test_authenticated_failure_still_falls_back_to_rss():
    """An expired token or a rate limit must not cost the source entirely."""
    recent = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"><title>r/x</title>
      <entry><title>via rss</title><link href="https://redd.it/1"/>
      <updated>{recent}</updated></entry>
    </feed>"""

    def fetch(url, **_kwargs):
        if OAUTH_HOST in url:
            raise ToolError("HTTP 401")
        return rss

    items = fetch_reddit(["LocalLLaMA"], token="stale", fetch=fetch)

    assert [i.title for i in items] == ["via rss"]


def test_without_a_token_the_dead_json_endpoint_is_not_even_tried():
    """Reddit 403s /hot.json for every unauthenticated client, on every host.
    Requesting it anyway would just add a guaranteed-failing round trip."""
    calls: list[str] = []
    recent = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    def fetch(url, **_kwargs):
        calls.append(url)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"><title>r/x</title>
          <entry><title>t</title><link href="https://redd.it/1"/>
          <updated>{recent}</updated></entry>
        </feed>"""

    fetch_reddit(["LocalLLaMA"], fetch=fetch)

    assert len(calls) == 1
    assert ".json" not in calls[0]
    assert ".rss" in calls[0]
