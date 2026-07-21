"""CLI entry point: `python -m radar.run`.

Wires the three sources to the loop, persists the outcome, and prints JSON.

The run row is opened *before* the agent starts and closed whatever happens, so
a failed pass leaves a record rather than a silence. When the daily job starts
misbehaving, the rows saying "failed, step_limit" are the ones worth reading.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from llm.calls_log import CallLog
from llm.client import AllProvidersFailed, LLMClient
from llm.config import NoProvidersConfigured, build_chain

from .agent import DEFAULT_MAX_STEPS, Agent, AgentFailed, RunResult
from .registry import ToolRegistry, ToolSpec
from .store import Store
from .tools.article import fetch_article_text
from .tools.hackernews import fetch_hackernews
from .tools.rss import fetch_rss

DEFAULT_GOAL = (
    "Find the 3-5 most relevant AI/LLM engineering developments of the last "
    "{hours} hours that are worth a technical post on a practitioner's blog. "
    "Cover both theoretical and practical angles where the evidence allows."
)

EXPECTED_FAILURES = (AgentFailed, AllProvidersFailed, NoProvidersConfigured)


def build_registry(*, max_items: int = 25) -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="fetch_hackernews",
                description=(
                    "Recent Hacker News stories. Omit `query` for everything in the window, "
                    "or pass one to search a specific topic."
                ),
                run=fetch_hackernews,
            ),
            ToolSpec(
                name="fetch_rss",
                description=(
                    "Latest posts from a curated list of AI blogs, newsletters and arXiv feeds."
                ),
                run=_fetch_rss_for_agent,
            ),
            ToolSpec(
                name="fetch_article_text",
                description=(
                    "Full readable text of one URL. Use it to check what a story actually "
                    "says before judging it."
                ),
                run=fetch_article_text,
            ),
        ],
        max_items=max_items,
    )


def _fetch_rss_for_agent(hours: int = 48):
    """The feed list is configuration, not something the model chooses."""
    return fetch_rss(hours=hours)


def run_agent(*, max_steps: int, max_items: int, db_path: str, goal: str) -> RunResult:
    load_dotenv()
    client = LLMClient(build_chain(os.environ), recorder=CallLog(db_path).record)
    agent = Agent(client, build_registry(max_items=max_items), max_steps=max_steps)
    return agent.run(goal)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="radar.run", description="Run one trend radar pass.")
    parser.add_argument("--hours", type=int, default=48, help="how far back to look")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--max-items", type=int, default=25, help="items per observation handed to the model"
    )
    parser.add_argument("--db", default="radar.db", help="history and call log")
    parser.add_argument("--export", default="", help="also write the history to this JSON file")
    parser.add_argument("--goal", default="", help="override the default goal")
    args = parser.parse_args(argv)

    goal = args.goal or DEFAULT_GOAL.format(hours=args.hours)
    store = Store(args.db)
    run_id = store.start_run(goal=goal, hours=args.hours)

    try:
        result = run_agent(
            max_steps=args.max_steps,
            max_items=args.max_items,
            db_path=args.db,
            goal=goal,
        )
    except EXPECTED_FAILURES as exc:
        # Expected operating conditions, not bugs. A traceback would bury the
        # one line saying what to fix.
        store.finish_run(run_id, status="failed", steps_used=0, stopped_because=type(exc).__name__)
        print(f"radar failed: {exc}", file=sys.stderr)
        return 1

    store.save_topics(run_id, result.topics)
    store.finish_run(
        run_id,
        status="ok",
        steps_used=result.steps_used,
        stopped_because=result.stopped_because,
    )
    if args.export:
        store.export_json(args.export)

    print(
        json.dumps(
            {
                "steps_used": result.steps_used,
                "stopped_because": result.stopped_because,
                "topics": [topic.as_dict() for topic in result.topics],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
