"""Hacker News via the Algolia search API — public, no auth, no rate key."""

from urllib.parse import urlencode

from ..errors import ToolError
from ..net import fetch_text
from .base import Fetcher, Item, cutoff, excerpt, parse_json_body, utc_from_epoch

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
ITEM_URL = "https://news.ycombinator.com/item?id={id}"


def fetch_hackernews(
    query: str | None = None,
    hours: int = 48,
    *,
    limit: int = 50,
    fetch: Fetcher = fetch_text,
) -> list[Item]:
    """Recent HN stories, newest first.

    The time window goes to the server as a numeric filter rather than being
    applied after the fact, so a busy day cannot push everything relevant off
    the end of the first page.
    """
    since = int(cutoff(hours))
    params = {
        "tags": "story",
        "numericFilters": f"created_at_i>{since}",
        "hitsPerPage": limit,
    }
    if query:
        params["query"] = query

    body = fetch(f"{SEARCH_URL}?{urlencode(params)}")
    data = parse_json_body(body, "hackernews")

    hits = data.get("hits")
    if not isinstance(hits, list):
        raise ToolError(f"hackernews response has no 'hits' list: {list(data)[:5]}")

    items = []
    for hit in hits:
        created = hit.get("created_at_i")
        if not isinstance(created, int | float) or created < since:
            continue
        items.append(
            Item(
                source="hackernews",
                title=hit.get("title") or "",
                # Ask HN and Show HN posts carry no external link.
                url=hit.get("url") or ITEM_URL.format(id=hit.get("objectID", "")),
                created_at=utc_from_epoch(created),
                author=hit.get("author") or "",
                score=int(hit.get("points") or 0),
                comments=int(hit.get("num_comments") or 0),
                text_excerpt=excerpt(hit.get("story_text") or ""),
            )
        )
    return items
