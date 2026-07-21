"""Reddit through its public endpoints — no OAuth.

Two paths, tried in order, because measured against the live site the obvious
one does not work:

1. `/hot.json` — rich: score, comment count, self-text. Returns 403 to
   unauthenticated clients on every host tried (www and old alike), so treat a
   success here as a bonus rather than the plan.
2. `old.reddit.com/r/<sub>/.rss` — still open. Titles and links only, no
   score, but a source with less metadata beats a source with no posts.

Each subreddit is independent: one blocked community costs that community and
nothing else. Only a clean sweep is worth an exception.
"""

import calendar
from datetime import datetime

import feedparser

from ..errors import ToolError
from ..net import fetch_text
from .base import Fetcher, Item, cutoff, excerpt, parse_json_body, utc_from_epoch

LISTING_URL = "https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
RSS_URL = "https://old.reddit.com/r/{subreddit}/.rss"
PERMALINK_BASE = "https://www.reddit.com"


def fetch_reddit(
    subreddits: list[str],
    hours: int = 48,
    *,
    limit: int = 50,
    fetch: Fetcher = fetch_text,
) -> list[Item]:
    since = cutoff(hours)
    items: list[Item] = []
    failures: list[str] = []

    for subreddit in subreddits:
        try:
            body = fetch(LISTING_URL.format(subreddit=subreddit, limit=limit))
            items.extend(_parse_listing(body, subreddit, since))
            continue
        except ToolError as exc:
            # Python unbinds the `except` name on exit, so keep a copy for the
            # failure message below.
            json_error = str(exc)

        try:
            body = fetch(RSS_URL.format(subreddit=subreddit))
            items.extend(_parse_rss(body, since))
        except ToolError as rss_error:
            failures.append(f"r/{subreddit}: json={json_error}; rss={rss_error}")

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
