import httpx
import pytest
import respx

from radar.errors import ToolError, UnsafeURL
from radar.net import check_url, fetch_text

PUBLIC = lambda _host: ["93.184.216.34"]  # noqa: E731 - a stand-in resolver for tests


# --- SSRF guard: the URLs reaching this function are chosen by an LLM reading
#     untrusted forum posts, so "looks fine" is not a security property. ---


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/api/chat",  # our own Ollama
        "http://localhost:8501",  # our own Streamlit
        "http://[::1]:80",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata service
        "http://10.0.0.5/admin",
        "http://192.168.1.1",
        "http://172.16.0.9",
        "http://0.0.0.0",
    ],
)
def test_private_and_loopback_addresses_are_refused(url):
    with pytest.raises(UnsafeURL):
        check_url(url, resolve=lambda host: [host.strip("[]")])


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com/x", "gopher://example.com", "data:text/html,hi"],
)
def test_non_http_schemes_are_refused(url):
    with pytest.raises(UnsafeURL):
        check_url(url, resolve=PUBLIC)


def test_url_without_a_host_is_refused():
    with pytest.raises(UnsafeURL):
        check_url("http:///just-a-path", resolve=PUBLIC)


def test_a_hostname_resolving_to_a_private_address_is_refused():
    """The dangerous case: a public-looking name pointed at an internal host."""
    with pytest.raises(UnsafeURL):
        check_url("https://totally-fine.example.com", resolve=lambda _h: ["10.1.2.3"])


def test_unresolvable_hostname_is_refused():
    def boom(_host):
        raise OSError("NXDOMAIN")

    with pytest.raises(UnsafeURL):
        check_url("https://nope.example.com", resolve=boom)


def test_public_https_url_is_allowed():
    check_url("https://news.ycombinator.com/item?id=1", resolve=PUBLIC)


# --- fetching ---


@respx.mock
def test_fetch_returns_body_text():
    respx.get("https://example.com/a").mock(
        return_value=httpx.Response(200, text="<html>hello</html>")
    )

    assert fetch_text("https://example.com/a", resolve=PUBLIC) == "<html>hello</html>"


@respx.mock
def test_oversized_response_is_truncated_not_swallowed_whole():
    respx.get("https://example.com/big").mock(return_value=httpx.Response(200, text="x" * 50_000))

    body = fetch_text("https://example.com/big", resolve=PUBLIC, max_bytes=1_000)

    assert len(body) <= 1_000


@respx.mock
def test_redirect_to_a_private_address_is_refused():
    """A public URL is not a promise about where it ends up."""
    respx.get("https://example.com/redir").mock(
        return_value=httpx.Response(302, headers={"Location": "http://169.254.169.254/"})
    )

    with pytest.raises(UnsafeURL):
        fetch_text(
            "https://example.com/redir",
            resolve=lambda host: ["93.184.216.34"] if "example" in host else [host],
        )


@respx.mock
def test_redirects_are_followed_up_to_a_limit():
    respx.get("https://example.com/1").mock(
        return_value=httpx.Response(302, headers={"Location": "https://example.com/2"})
    )
    respx.get("https://example.com/2").mock(return_value=httpx.Response(200, text="arrived"))

    assert fetch_text("https://example.com/1", resolve=PUBLIC) == "arrived"


@respx.mock
def test_redirect_loop_gives_up():
    respx.get("https://example.com/loop").mock(
        return_value=httpx.Response(302, headers={"Location": "https://example.com/loop"})
    )

    with pytest.raises(ToolError):
        fetch_text("https://example.com/loop", resolve=PUBLIC, max_redirects=3)


@respx.mock
def test_http_error_becomes_a_tool_error():
    respx.get("https://example.com/gone").mock(return_value=httpx.Response(404))

    with pytest.raises(ToolError):
        fetch_text("https://example.com/gone", resolve=PUBLIC)


@respx.mock
def test_network_failure_becomes_a_tool_error():
    respx.get("https://example.com/down").mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(ToolError):
        fetch_text("https://example.com/down", resolve=PUBLIC)


@respx.mock
def test_requests_identify_themselves():
    """Anonymous scrapers get blocked, and rightly so."""
    route = respx.get("https://example.com/ua").mock(return_value=httpx.Response(200, text="ok"))

    fetch_text("https://example.com/ua", resolve=PUBLIC)

    assert "ai-blog-copilot" in route.calls.last.request.headers["user-agent"]
