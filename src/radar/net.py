"""The only place in the radar that opens a socket.

Everything here exists because of one uncomfortable fact: the URLs this module
fetches are chosen by an LLM that has just read attacker-controlled text from
Hacker News and a dozen syndicated blogs. A URL appearing in the model's output
says nothing
about whether it is safe to request.

So every request is checked before it leaves the machine, and — the part that
is easy to forget — every redirect hop is checked again. A public hostname is
not a promise about where the response comes from.
"""

import ipaddress
import socket
from collections.abc import Callable, Iterable
from urllib.parse import urljoin, urlparse

import httpx

from .errors import ToolError, UnsafeURL

USER_AGENT = (
    "ai-blog-copilot/0.1 (+https://github.com/lepablito/ai-blog-copilot) personal trend-radar bot"
)

DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_MAX_REDIRECTS = 5

Resolver = Callable[[str], Iterable[str]]


def _default_resolve(host: str) -> list[str]:
    """Every address the host resolves to — all of them must be public."""
    infos = socket.getaddrinfo(host, None)
    return [info[4][0] for info in infos]


def check_url(url: str, *, resolve: Resolver = _default_resolve) -> None:
    """Raise `UnsafeURL` unless this URL is safe to request.

    Refuses non-HTTP schemes, and any host that resolves to an address outside
    the public internet: loopback, RFC1918, link-local (which is how cloud
    metadata services at 169.254.169.254 get reached), and friends.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise UnsafeURL(f"refusing non-HTTP scheme {parsed.scheme!r} in {url!r}")

    host = parsed.hostname
    if not host:
        raise UnsafeURL(f"no host in {url!r}")

    try:
        addresses = list(resolve(host))
    except OSError as exc:
        raise UnsafeURL(f"cannot resolve {host!r}: {exc}") from exc

    if not addresses:
        raise UnsafeURL(f"{host!r} resolved to nothing")

    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise UnsafeURL(f"{host!r} resolved to an unparseable address {address!r}") from exc

        if not ip.is_global:
            raise UnsafeURL(f"{host!r} resolves to non-public address {address}")


def fetch_text(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    resolve: Resolver = _default_resolve,
    headers: dict[str, str] | None = None,
) -> str:
    """GET `url` and return its body as text, capped at `max_bytes`.

    Redirects are followed manually rather than by httpx, because each hop has
    to pass `check_url` before it is requested.
    """
    request_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    current = url

    for _hop in range(max_redirects + 1):
        check_url(current, resolve=resolve)

        try:
            with (
                httpx.Client(follow_redirects=False, timeout=timeout) as client,
                client.stream("GET", current, headers=request_headers) as response,
            ):
                if response.is_redirect:
                    location = response.headers.get("Location")
                    if not location:
                        raise ToolError(f"redirect without a Location header from {current!r}")
                    current = urljoin(current, location)
                    continue

                if response.status_code >= 400:
                    raise ToolError(f"HTTP {response.status_code} from {current!r}")

                body = _read_capped(response, max_bytes)
        except httpx.TransportError as exc:
            raise ToolError(f"network failure for {current!r}: {exc}") from exc

        return body

    raise ToolError(f"more than {max_redirects} redirects starting from {url!r}")


def _read_capped(response: httpx.Response, max_bytes: int) -> str:
    """Stop reading at `max_bytes` — a hostile or broken URL should not be able
    to exhaust memory just by serving an endless body."""
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_bytes:
            break

    raw = b"".join(chunks)[:max_bytes]
    return raw.decode(response.encoding or "utf-8", errors="replace")
