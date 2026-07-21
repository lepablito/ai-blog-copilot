"""Full text of a single page, for when a headline is not enough to judge it.

This is the tool the agent points at URLs it found inside untrusted content, so
it is also the tool most likely to be aimed somewhere it should not go. It does
no fetching of its own: everything goes through `net.fetch_text`, which
enforces the SSRF guard on the URL and on every redirect after it.
"""

import trafilatura

from ..errors import ToolError
from ..net import fetch_text
from .base import Fetcher

DEFAULT_LIMIT = 8_000

# Extraction on a nav-only page happily returns "menu". Anything under this is
# scaffolding, not prose — and a stub excerpt is worse than an honest failure,
# because the model will summarise it as though it were the article.
MIN_CHARS = 200


def fetch_article_text(
    url: str,
    *,
    limit: int = DEFAULT_LIMIT,
    fetch: Fetcher = fetch_text,
) -> str:
    """Return the readable body of `url`, stripped of navigation and footers.

    Raises ToolError when the page yields nothing worth reading — a paywall, a
    JS-only shell, or a redirect to a cookie wall. Returning an empty string
    instead would quietly poison the model's summary.
    """
    html = fetch(url)

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    text = (text or "").strip()
    if len(text) < MIN_CHARS:
        raise ToolError(
            f"no readable article text at {url!r} — extracted {len(text)} chars "
            "(paywall, cookie wall or JS-only page?)"
        )

    return text[:limit]
