import time
from datetime import UTC, datetime
from email.utils import formatdate

import pytest

from radar.errors import ToolError
from radar.tools.article import fetch_article_text
from radar.tools.rss import fetch_rss, load_feeds

HOUR = 3600


def rss_feed(entries):
    body = "".join(
        f"""
        <item>
          <title>{e['title']}</title>
          <link>{e['link']}</link>
          <pubDate>{formatdate(e['when'])}</pubDate>
          <description>{e.get('summary', 'a summary')}</description>
        </item>
        """
        for e in entries
    )
    return f"""<?xml version="1.0"?>
    <rss version="2.0"><channel><title>Test feed</title>{body}</channel></rss>"""


def write_feeds(tmp_path, feeds):
    path = tmp_path / "feeds.yaml"
    lines = "\n".join(f'  - name: "{n}"\n    url: "{u}"' for n, u in feeds)
    path.write_text(f"feeds:\n{lines}\n", encoding="utf-8")
    return path


# --- feeds.yaml ---


def test_feeds_are_loaded_from_yaml(tmp_path):
    path = write_feeds(tmp_path, [("Anthropic", "https://a.example/rss")])

    feeds = load_feeds(path)

    assert feeds == [{"name": "Anthropic", "url": "https://a.example/rss"}]


def test_missing_feeds_file_raises_tool_error(tmp_path):
    with pytest.raises(ToolError):
        load_feeds(tmp_path / "nope.yaml")


def test_feed_entry_without_a_url_is_rejected(tmp_path):
    path = tmp_path / "feeds.yaml"
    path.write_text('feeds:\n  - name: "broken"\n', encoding="utf-8")

    with pytest.raises(ToolError):
        load_feeds(path)


def test_the_shipped_feeds_file_is_valid():
    """The default feed list is configuration people edit — keep it parseable."""
    feeds = load_feeds("feeds.yaml")

    assert len(feeds) >= 5
    assert all(f["url"].startswith("http") for f in feeds)


# --- fetching feeds ---


def test_rss_entries_become_items(tmp_path):
    path = write_feeds(tmp_path, [("Blog", "https://b.example/rss")])
    body = rss_feed([{"title": "New model", "link": "https://b.example/1", "when": time.time()}])

    items = fetch_rss(feeds_from=path, fetch=lambda _url, **_kw: body)

    (item,) = items
    assert item.source == "rss"
    assert item.title == "New model"
    assert item.url == "https://b.example/1"
    assert item.author == "Blog"


def test_rss_drops_entries_older_than_the_window(tmp_path):
    path = write_feeds(tmp_path, [("Blog", "https://b.example/rss")])
    body = rss_feed(
        [
            {"title": "old", "link": "https://b.example/old", "when": time.time() - 200 * HOUR},
            {"title": "fresh", "link": "https://b.example/new", "when": time.time() - HOUR},
        ]
    )

    items = fetch_rss(feeds_from=path, hours=48, fetch=lambda _url, **_kw: body)

    assert [i.title for i in items] == ["fresh"]


def test_one_broken_feed_does_not_lose_the_others(tmp_path):
    path = write_feeds(
        tmp_path, [("Broken", "https://broken.example/rss"), ("Good", "https://good.example/rss")]
    )
    good = rss_feed([{"title": "survived", "link": "https://good.example/1", "when": time.time()}])

    def fetch(url, **_kwargs):
        if "broken" in url:
            raise ToolError("HTTP 500")
        return good

    items = fetch_rss(feeds_from=path, fetch=fetch)

    assert [i.title for i in items] == ["survived"]


def test_unparseable_feed_is_skipped_not_fatal(tmp_path):
    path = write_feeds(
        tmp_path, [("Junk", "https://junk.example/rss"), ("Good", "https://good.example/rss")]
    )
    good = rss_feed([{"title": "survived", "link": "https://good.example/1", "when": time.time()}])

    def fetch(url, **_kwargs):
        return "this is not XML at all" if "junk" in url else good

    items = fetch_rss(feeds_from=path, fetch=fetch)

    assert [i.title for i in items] == ["survived"]


def test_absurd_publication_date_is_skipped_not_crashed(tmp_path):
    """A real feed in the wild carries a year-9999 date. It must not end the run."""
    path = write_feeds(tmp_path, [("Blog", "https://b.example/rss")])
    body = f"""<?xml version="1.0"?>
    <rss version="2.0"><channel><title>t</title>
      <item><title>bogus</title><link>https://b.example/bogus</link>
        <pubDate>Fri, 01 Jan 9999 00:00:00 GMT</pubDate></item>
      <item><title>fine</title><link>https://b.example/fine</link>
        <pubDate>{formatdate(time.time())}</pubDate></item>
    </channel></rss>"""

    items = fetch_rss(feeds_from=path, fetch=lambda _url, **_kw: body)

    assert [i.title for i in items] == ["fine"]


def test_publication_dates_are_read_as_utc(tmp_path):
    """feedparser hands back UTC. Reading it as local time silently shifts every
    entry by the machine's offset — and the whole point is a 48-hour window."""
    path = write_feeds(tmp_path, [("Blog", "https://b.example/rss")])
    body = """<?xml version="1.0"?>
    <rss version="2.0"><channel><title>t</title>
      <item><title>noon</title><link>https://b.example/noon</link>
        <pubDate>Wed, 01 Jan 2025 12:00:00 GMT</pubDate></item>
    </channel></rss>"""

    items = fetch_rss(feeds_from=path, hours=24 * 365 * 10, fetch=lambda _url, **_kw: body)

    assert items[0].created_at == datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def test_every_feed_failing_raises_tool_error(tmp_path):
    path = write_feeds(tmp_path, [("A", "https://a.example/rss")])

    def fetch(_url, **_kwargs):
        raise ToolError("HTTP 500")

    with pytest.raises(ToolError):
        fetch_rss(feeds_from=path, fetch=fetch)


# --- article extraction ---

ARTICLE_HTML = """
<html><head><title>Why agents fail</title></head><body>
<nav>home about contact</nav>
<article>
  <h1>Why agents fail</h1>
  <p>The first paragraph explains that most agent demos never survive contact
  with a real tool that returns an error instead of a tidy answer.</p>
  <p>The second paragraph goes into retries, budgets and step limits in far
  more detail than any navigation menu ever could.</p>
</article>
<footer>copyright 2026</footer>
</body></html>
"""


def test_article_text_is_extracted_without_boilerplate():
    text = fetch_article_text("https://example.com/a", fetch=lambda _url, **_kw: ARTICLE_HTML)

    assert "most agent demos never survive" in text
    assert "copyright 2026" not in text


def test_article_text_is_capped():
    html = f"<html><body><article><p>{'word ' * 50_000}</p></article></body></html>"

    text = fetch_article_text("https://example.com/a", fetch=lambda _url, **_kw: html, limit=500)

    assert len(text) <= 500


def test_page_with_no_extractable_content_raises_tool_error():
    def fetch(_url, **_kwargs):
        return "<html><body><nav>menu</nav></body></html>"

    with pytest.raises(ToolError):
        fetch_article_text("https://example.com/empty", fetch=fetch)


def test_fetch_failure_propagates_as_tool_error():
    def fetch(_url, **_kwargs):
        raise ToolError("HTTP 404")

    with pytest.raises(ToolError):
        fetch_article_text("https://example.com/gone", fetch=fetch)
