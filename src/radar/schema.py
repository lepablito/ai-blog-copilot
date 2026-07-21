"""The agent's output contract.

Validation errors are written to be read *by the model*: they name the field
and the offending topic, because the agent feeds them straight back as a repair
prompt. "invalid input" would leave it guessing.
"""

from dataclasses import dataclass, field
from typing import Any

ANGLES = ("theoretical", "practical")

REQUIRED_TEXT = ("title", "summary", "why_now")


class InvalidTopics(Exception):
    """The final answer did not match the contract."""


@dataclass(slots=True)
class Topic:
    title: str
    summary: str
    sources: list[str]
    why_now: str
    angle: str
    suggested_outline: list[str]
    estimated_effort: str = "unknown"
    citations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sources": self.sources,
            "why_now": self.why_now,
            "angle": self.angle,
            "estimated_effort": self.estimated_effort,
            "suggested_outline": self.suggested_outline,
            "citations": self.citations,
        }


def parse_topics(payload: Any) -> list[Topic]:
    raw = _topic_list(payload)
    if not raw:
        raise InvalidTopics("no topics were returned — expected between 3 and 5")

    return [_one_topic(entry, position) for position, entry in enumerate(raw, start=1)]


def _topic_list(payload: Any) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        topics = payload.get("topics")
        if isinstance(topics, list):
            return topics
        raise InvalidTopics(
            f"expected an object with a 'topics' array, got keys {list(payload)[:6]}"
        )
    raise InvalidTopics(f"expected a JSON object or array, got {type(payload).__name__}")


def _one_topic(entry: Any, position: int) -> Topic:
    where = f"topic {position}"
    if not isinstance(entry, dict):
        raise InvalidTopics(f"{where}: expected an object, got {type(entry).__name__}")

    for name in REQUIRED_TEXT:
        value = entry.get(name)
        if not isinstance(value, str) or not value.strip():
            raise InvalidTopics(f"{where}: field '{name}' must be a non-empty string")

    angle = str(entry.get("angle") or "").strip().lower()
    if angle not in ANGLES:
        raise InvalidTopics(f"{where}: field 'angle' must be one of {ANGLES}, got {angle!r}")

    sources = _url_list(entry.get("sources"), "sources", where)
    if not sources:
        raise InvalidTopics(f"{where}: field 'sources' needs at least one http(s) URL")

    outline = entry.get("suggested_outline")
    if not isinstance(outline, list) or not [b for b in outline if str(b).strip()]:
        raise InvalidTopics(f"{where}: field 'suggested_outline' needs at least one bullet")

    return Topic(
        title=entry["title"].strip(),
        summary=entry["summary"].strip(),
        sources=sources,
        why_now=entry["why_now"].strip(),
        angle=angle,
        suggested_outline=[str(b).strip() for b in outline if str(b).strip()],
        estimated_effort=str(entry.get("estimated_effort") or "unknown").strip() or "unknown",
        citations=_url_list(entry.get("citations"), "citations", where),
    )


def _url_list(value: Any, name: str, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise InvalidTopics(f"{where}: field '{name}' must be an array of URLs")

    urls = []
    for candidate in value:
        text = str(candidate).strip()
        if not text:
            continue
        # Anything that is not http(s) is either a hallucinated citation or a
        # scheme we would refuse to fetch anyway.
        if not text.startswith(("http://", "https://")):
            raise InvalidTopics(f"{where}: field '{name}' contains a non-http URL: {text[:60]!r}")
        urls.append(text)
    return urls
