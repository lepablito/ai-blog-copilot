"""A ReAct loop, written out by hand.

Thought → Action → Observation, with no agent framework underneath. The whole
point of building it this way is that the failure handling is visible rather
than inherited, and every one of these cases is load-bearing:

* The model returns something that is not a legal move → it gets told, and the
  loop continues.
* A tool does not exist, or blows up → an ERROR observation, and the loop
  continues. Containment lives in the registry.
* Steps run out → one final call that forbids tools and demands an answer from
  the evidence already gathered. Spending a full run's tokens and returning
  nothing is the worst available outcome.
* The final answer fails validation → the schema error goes back as a repair
  prompt, once. Twice and it raises, because a model that cannot produce the
  contract twice will not produce it on the fifth attempt either.
"""

from dataclasses import dataclass, field
from typing import Any

from llm.base import Message
from llm.client import LLMClient

from . import prompts
from .registry import ToolRegistry
from .sanitize import new_nonce
from .schema import InvalidTopics, Topic, parse_topics

DEFAULT_MAX_STEPS = 8

# Measured, not guessed: at 2048 a live run truncated the final_answer mid-JSON
# and burned an extra call regenerating it. Five topics with outlines and
# citations land around 2-3k output tokens, so this leaves real headroom.
DEFAULT_MAX_TOKENS = 6144


class AgentFailed(Exception):
    """The run produced no usable topics."""


@dataclass(slots=True)
class Step:
    thought: str
    tool: str | None
    args: dict | None
    observation: str


@dataclass(slots=True)
class RunResult:
    topics: list[Topic]
    steps_used: int
    stopped_because: str
    transcript: list[Step] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        nonce: str | None = None,
    ):
        self._client = client
        self._registry = registry
        self._max_steps = max_steps
        self._max_tokens = max_tokens
        self._nonce = nonce or new_nonce()

    def _ask(self, messages: list[Message], purpose: str) -> Any:
        return self._client.generate_json(
            messages, max_tokens=self._max_tokens, purpose=purpose
        )

    def run(self, goal: str) -> RunResult:
        messages: list[Message] = [
            {
                "role": "system",
                "content": prompts.SYSTEM.format(
                    tools=self._registry.describe(), nonce=self._nonce
                ),
            },
            {
                "role": "user",
                "content": prompts.GOAL.format(goal=goal, max_steps=self._max_steps),
            },
        ]

        transcript: list[Step] = []

        for step in range(1, self._max_steps + 1):
            reply = self._ask(messages, "radar:step")
            thought = str(reply.get("thought") or "") if isinstance(reply, dict) else ""

            if isinstance(reply, dict) and "final_answer" in reply:
                topics = self._finalise(reply["final_answer"], messages)
                transcript.append(Step(thought, None, None, "final answer accepted"))
                return RunResult(topics, step, "final_answer", transcript)

            observation, tool, args = self._act(reply)
            transcript.append(Step(thought, tool, args, observation))
            messages.append({"role": "assistant", "content": _as_text(reply)})
            messages.append({"role": "user", "content": observation})

        # Out of steps. One last call, tools forbidden.
        messages.append(
            {"role": "user", "content": prompts.CLOSING.format(max_steps=self._max_steps)}
        )
        reply = self._ask(messages, "radar:closing")
        answer = reply.get("final_answer", reply) if isinstance(reply, dict) else reply
        topics = self._finalise(answer, messages)
        transcript.append(Step("forced close", None, None, "final answer accepted"))
        return RunResult(topics, self._max_steps, "step_limit", transcript)

    def _act(self, reply: Any) -> tuple[str, str | None, dict | None]:
        if not isinstance(reply, dict) or "action" not in reply:
            return prompts.NO_ACTION, None, None

        act = reply["action"]
        if not isinstance(act, dict) or not act.get("tool"):
            return (
                "ERROR: 'action' must be an object with a 'tool' name and an 'args' object.",
                None,
                None,
            )

        tool = str(act["tool"])
        args = act.get("args", {})
        observation = self._registry.call(tool, args, nonce=self._nonce)
        return observation, tool, args if isinstance(args, dict) else None

    def _finalise(self, answer: Any, messages: list[Message]) -> list[Topic]:
        try:
            return parse_topics(answer)
        except InvalidTopics as first_error:
            messages.append({"role": "assistant", "content": _as_text(answer)})
            messages.append(
                {"role": "user", "content": prompts.REPAIR.format(error=first_error)}
            )
            repaired = self._ask(messages, "radar:repair")
            candidate = (
                repaired.get("final_answer", repaired)
                if isinstance(repaired, dict)
                else repaired
            )
            try:
                return parse_topics(candidate)
            except InvalidTopics as second_error:
                raise AgentFailed(
                    f"final answer failed validation twice: {second_error}"
                ) from second_error


def _as_text(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)

