"""Curated RSS/Atom feeds, listed in feeds.yaml.

feedparser is deliberately forgiving about malformed XML, which is the right
default for the open web but means "it parsed" is not the same as "it worked".
A feed that yields no entries is skipped rather than trusted, and — as with
Reddit — only a total wipeout is worth raising.
"""

import calendar
from datetime import datetime
from pathlib import Path

import feedparser
import yaml

from ..errors import ToolError
from ..net import fetch_text
from .base import Fetcher, Item, cutoff, excerpt, utc_from_epoch

DEFAULT_FEEDS_FILE = "feeds.yaml"


def load_feeds(path: Path | str = DEFAULT_FEEDS_FILE) -> list[dict[str, str]]:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ToolError(f"cannot read feed list {str(path)!r}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ToolError(f"feed list {str(path)!r} is not valid YAML: {exc}") from exc

    feeds = (raw or {}).get("feeds")
    if not isinstance(feeds, list) or not feeds:
        raise ToolError(f"feed list {str(path)!r} has no 'feeds' entries")

    for feed in feeds:
        if not isinstance(feed, dict) or not feed.get("url"):
            raise ToolError(f"feed entry without a url in {str(path)!r}: {feed!r}")

    return [{"name": f.get("name") or f["url"], "url": f["url"]} for f in feeds]


def fetch_rss(
    feeds_from: Path | str = DEFAULT_FEEDS_FILE,
    hours: int = 48,
    *,
    fetch: Fetcher = fetch_text,
) -> list[Item]:
    feeds = load_feeds(feeds_from)
    since = cutoff(hours)

    items: list[Item] = []
    failures: list[str] = []

    for feed in feeds:
        try:
            body = fetch(feed["url"])
        except ToolError as exc:
            failures.append(f"{feed['name']}: {exc}")
            continue

        parsed = feedparser.parse(body)
        entries = parsed.get("entries") or []
        if not entries:
            failures.append(f"{feed['name']}: no entries (feed may be malformed)")
            continue

        items.extend(_to_items(entries, feed["name"], since))

    if failures and not items:
        raise ToolError("every feed failed — " + "; ".join(failures))

    return items


def _to_items(entries: list, feed_name: str, since: float) -> list[Item]:
    items = []
    for entry in entries:
        published = _published_at(entry)
        if published is None or published.timestamp() < since:
            continue

        link = entry.get("link") or ""
        if not link:
            continue

        items.append(
            Item(
                source="rss",
                title=entry.get("title") or "",
                url=link,
                created_at=published,
                author=feed_name,
                text_excerpt=excerpt(entry.get("summary") or ""),
            )
        )
    return items


def _published_at(entry) -> datetime | None:
    """Feeds disagree on which date field they populate; try both.

    Two traps here, both found against live feeds rather than fixtures:

    * feedparser normalises `*_parsed` to UTC, so `time.mktime` — which reads a
      struct as *local* time — shifts every entry by the machine's offset. On a
      48-hour window that is a real distortion. `calendar.timegm` is the UTC
      counterpart.
    * Feeds in the wild carry dates like year 9999. `timegm` is pure arithmetic
      and returns them happily; the conversion to a datetime is what blows up.
      So the two steps stay together and one unusable date costs one entry.
    """
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if not parsed:
            continue
        try:
            return utc_from_epoch(calendar.timegm(parsed))
        except (OverflowError, ValueError, OSError):
            return None
    return None
