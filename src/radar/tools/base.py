"""The shape every source collapses into.

Hacker News, Reddit and RSS disagree about almost everything — field names,
timestamp formats, what counts as a score. Normalising at the edge means the
agent loop reasons about one kind of thing, and adding a fourth source later
costs one adapter rather than a change to the prompt.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from ..errors import ToolError

# A tool asks for a URL and gets text back. Injecting this is what lets every
# tool test run without a socket.
Fetcher = Callable[..., str]


@dataclass(slots=True)
class Item:
    source: str
    title: str
    url: str
    created_at: datetime | None = None
    author: str = ""
    score: int = 0
    comments: int = 0
    text_excerpt: str = ""

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "author": self.author,
            "score": self.score,
            "comments": self.comments,
            "text_excerpt": self.text_excerpt,
        }


EXCERPT_CHARS = 1_200


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """Trim a body to something a prompt can afford.

    Sanitising of this text happens later, in one place, so that nothing can
    slip into a prompt unwrapped.
    """
    collapsed = " ".join((text or "").split())
    return collapsed[:limit]


def parse_json_body(body: str, source: str) -> dict:
    """Decode an API response, turning every failure into a ToolError.

    A rate-limit page or a Cloudflare challenge arrives as HTML with a 200, so
    "it parsed as JSON" is the only real success signal here.
    """
    try:
        data = json.loads(body)
    except ValueError as exc:
        raise ToolError(f"{source} returned a non-JSON body: {body[:200]!r}") from exc
    if not isinstance(data, dict):
        raise ToolError(f"{source} returned {type(data).__name__}, expected an object")
    return data


def utc_from_epoch(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, tz=UTC)


def cutoff(hours: int) -> float:
    return datetime.now(UTC).timestamp() - hours * 3600
