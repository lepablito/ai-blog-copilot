"""Tool dispatch: the boundary between the model's intentions and real code.

Two jobs, both about containment.

**Nothing here raises.** A hallucinated tool name, a made-up argument, a dead
API, an outright bug in a tool — all of them come back as an ERROR observation
the model can read and route around. A ReAct loop that dies on the first bad
tool call is a demo, not a system.

**Everything that came off the network gets wrapped.** Tool output is fenced by
`sanitize` here rather than at each call site, so there is one place to audit
and no way for a future tool to forget.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import ToolError
from .sanitize import render_items, wrap_untrusted
from .tools.base import Item

DEFAULT_MAX_ITEMS = 25


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    run: Callable[..., Any]

    def signature(self) -> str:
        """Rendered into the system prompt — this is how the model learns what
        it is allowed to pass."""
        rendered = []
        for parameter in inspect.signature(self.run).parameters.values():
            annotation = (
                parameter.annotation.__name__ if isinstance(parameter.annotation, type) else "any"
            )
            has_default = parameter.default is not inspect.Parameter.empty
            default = f" = {parameter.default!r}" if has_default else ""
            rendered.append(f"{parameter.name}: {annotation}{default}")
        return ", ".join(rendered)


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec], *, max_items: int = DEFAULT_MAX_ITEMS):
        self._tools = {tool.name: tool for tool in tools}
        self._max_items = max_items

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        return "\n".join(
            f"- {tool.name}({tool.signature()}) — {tool.description}"
            for tool in self._tools.values()
        )

    def call(self, name: str, arguments: Any, *, nonce: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return _error(f"unknown tool {name!r}. Available tools: {', '.join(self._tools)}")

        if not isinstance(arguments, dict):
            return _error(
                f"'args' for {name!r} must be a JSON object, got {type(arguments).__name__}"
            )

        try:
            inspect.signature(tool.run).bind(**arguments)
        except TypeError as exc:
            return _error(f"bad arguments for {name!r}: {exc}")

        try:
            result = tool.run(**arguments)
        except ToolError as exc:
            return _error(f"{name} failed: {exc}")
        except Exception as exc:  # noqa: BLE001 - a tool bug must not end the run
            return _error(f"{name} crashed: {type(exc).__name__}: {exc}")

        return self._render(result, nonce=nonce, source=name)

    def _render(self, result: Any, *, nonce: str, source: str) -> str:
        if isinstance(result, str):
            return wrap_untrusted(result, nonce=nonce, source=source)

        items = list(result or [])
        shown = _most_promising(items, self._max_items)
        rendered = render_items(shown, nonce=nonce, source=source)

        if len(items) > len(shown):
            # Say so explicitly: a silently truncated list invites the model to
            # conclude the window was quiet when it was the opposite.
            rendered = (
                f"Showing the {len(shown)} highest-signal of {len(items)} items found.\n"
                + rendered
            )
        return rendered


def _most_promising(items: list[Item], limit: int) -> list[Item]:
    """Rank by traction, then recency.

    Twelve feeds over 48 hours run to hundreds of items — arXiv alone floods
    it. Handing all of them to the model would blow the context and bury the
    signal, so the cut happens here, on measured numbers, rather than asking
    the model to skim.
    """
    return sorted(
        items,
        key=lambda i: (i.score, i.comments, i.created_at.timestamp() if i.created_at else 0),
        reverse=True,
    )[:limit]


def _error(message: str) -> str:
    """Error text is ours, not the network's, so it stays outside the fence —
    it is one of the few things in the loop the model *should* act on."""
    return f"ERROR: {message}"
