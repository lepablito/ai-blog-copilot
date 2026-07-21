"""Reddit, with and without credentials.

Measured against the live site rather than assumed: the unauthenticated
`/hot.json` endpoint returns 403 for every User-Agent tried, browser strings
included, on both www and old. It is not rate limiting, it is policy, so the
tool does not waste a round trip on it.

That leaves two paths:

1. **Authenticated** — an app-only OAuth token against `oauth.reddit.com`
   returns the full listing: score, comment count, self-text. This is what
   makes Reddit rankable; without a score the agent cannot tell a post with two
   thousand upvotes from one with three.
2. **`old.reddit.com/r/<sub>/.rss`** — still open to anyone. Titles and links
   only, and used both as the no-credentials path and as the fallback when a
   token expires mid-run.

Each subreddit is independent: one blocked community costs that community and
nothing else. Only a clean sweep is worth an exception.
"""

import calendar
from collections.abc import Mapping
from datetime import datetime

import feedparser

from ..errors import ToolError
from ..net import fetch_text, post_form
from .base import Fetcher, Item, cutoff, excerpt, parse_json_body, utc_from_epoch

OAUTH_HOST = "oauth.reddit.com"
TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
OAUTH_LISTING_URL = "https://oauth.reddit.com/r/{subreddit}/hot?limit={limit}"
RSS_URL = "https://old.reddit.com/r/{subreddit}/.rss"
PERMALINK_BASE = "https://www.reddit.com"


def get_app_token(client_id: str, client_secret: str, *, post=post_form) -> str:
    """Exchange script-app credentials for an app-only bearer token.

    Reddit tokens last around 24 hours — far longer than a radar run — so this
    is called once per run and the token passed down. No cache, no expiry
    bookkeeping, nothing to go stale between runs.
    """
    body = post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
    )

    payload = parse_json_body(body, "reddit token")
    token = payload.get("access_token")
    if not token:
        raise ToolError(f"reddit token response carried no access_token: {list(payload)[:5]}")
    return token


def token_from_env(env: Mapping[str, str], *, get_token=get_app_token) -> str | None:
    """Best-effort token. Returns None rather than raising.

    Reddit credentials are an enhancement, not a dependency: without them the
    radar still runs on the RSS listing. A typo in the secret should cost the
    score column, not the day's run.
    """
    client_id = (env.get("REDDIT_CLIENT_ID") or "").strip()
    client_secret = (env.get("REDDIT_CLIENT_SECRET") or "").strip()
    if not client_id or not client_secret:
        return None

    try:
        return get_token(client_id, client_secret)
    except ToolError:
        return None


def fetch_reddit(
    subreddits: list[str],
    hours: int = 48,
    *,
    limit: int = 50,
    token: str | None = None,
    fetch: Fetcher = fetch_text,
) -> list[Item]:
    since = cutoff(hours)
    items: list[Item] = []
    failures: list[str] = []

    for subreddit in subreddits:
        oauth_error = None

        if token:
            try:
                body = fetch(
                    OAUTH_LISTING_URL.format(subreddit=subreddit, limit=limit),
                    headers={"Authorization": f"Bearer {token}"},
                )
                items.extend(_parse_listing(body, subreddit, since))
                continue
            except ToolError as exc:
                # Python unbinds the `except` name on exit — keep a copy.
                oauth_error = str(exc)

        try:
            body = fetch(RSS_URL.format(subreddit=subreddit))
            items.extend(_parse_rss(body, since))
        except ToolError as rss_error:
            detail = f"oauth={oauth_error}; rss={rss_error}" if oauth_error else str(rss_error)
            failures.append(f"r/{subreddit}: {detail}")

    if failures and not items:
        raise ToolError("every subreddit failed — " + "; ".join(failures))

    return items


def _parse_listing(body: str, subreddit: str, since: float) -> list[Item]:
    data = parse_json_body(body, f"reddit r/{subreddit}")
    children = (data.get("data") or {}).get("children")
    if not isinstance(children, list):
        raise ToolError(f"r/{subreddit} response has no listing")

    items = []
    for child in children:
        post = child.get("data") or {}

        # Pinned mod announcements are permanent fixtures, never news.
        if post.get("stickied"):
            continue

        created = post.get("created_utc")
        if not isinstance(created, int | float) or created < since:
            continue

        permalink = post.get("permalink") or ""
        items.append(
            Item(
                source="reddit",
                title=post.get("title") or "",
                url=post.get("url") or f"{PERMALINK_BASE}{permalink}",
                created_at=utc_from_epoch(created),
                author=post.get("author") or "",
                score=int(post.get("score") or 0),
                comments=int(post.get("num_comments") or 0),
                text_excerpt=excerpt(post.get("selftext") or ""),
            )
        )
    return items


def _parse_rss(body: str, since: float) -> list[Item]:
    """The degraded path: titles and links, and deliberately no score.

    Leaving score at 0 rather than guessing keeps the agent from ranking a
    fallback-sourced post against a real one on a number nobody measured.
    """
    parsed = feedparser.parse(body)
    entries = parsed.get("entries") or []
    if not entries:
        raise ToolError("rss listing had no entries")

    items = []
    for entry in entries:
        published = _entry_datetime(entry)
        if published is None or published.timestamp() < since:
            continue

        link = entry.get("link") or ""
        if not link:
            continue

        items.append(
            Item(
                source="reddit",
                title=entry.get("title") or "",
                url=link,
                created_at=published,
                author=(entry.get("author") or "").removeprefix("/u/"),
                text_excerpt=excerpt(entry.get("summary") or ""),
            )
        )
    return items


def _entry_datetime(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if not parsed:
            continue
        try:
            return utc_from_epoch(calendar.timegm(parsed))
        except (OverflowError, ValueError, OSError):
            return None
    return None
