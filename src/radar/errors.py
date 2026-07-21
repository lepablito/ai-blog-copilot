"""Failure vocabulary for the radar tools.

Tools never leak raw httpx or parser exceptions. The agent loop needs exactly
one thing from a failing tool: a message it can hand back to the model as an
observation so the run continues with one fewer source.
"""


class ToolError(Exception):
    """A tool could not do its job. Recoverable — the loop keeps going."""


class UnsafeURL(ToolError):
    """A URL was refused before any request left the machine."""
