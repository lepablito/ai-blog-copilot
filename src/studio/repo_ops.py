"""Starting a radar run and bringing its results back.

No Streamlit imports here on purpose, same as `radar_data`: the view decides
what to draw, this module decides what to run and what the result means.

Both commands go out as fixed argument lists with no shell, and nothing from
the UI or from a model is ever interpolated into them. `runner` is injected so
tests can drive every outcome without a workflow run or a network round trip.
"""

import subprocess
from collections.abc import Callable
from dataclasses import dataclass

WORKFLOW = "daily-radar.yml"

# `gh workflow run` prints nothing identifying the run it started, so the link
# offered afterwards is to the workflow's own page rather than to that run.
ACTIONS_URL = "https://github.com/lepablito/ai-blog-copilot/actions/workflows/daily-radar.yml"

# Long enough for git to reach GitHub on a slow line, short enough that a stuck
# process does not leave the tab hanging with no way back — Streamlit runs this
# on the thread drawing the page.
TIMEOUT_SECONDS = 60

MISSING = {
    "gh": "gh is not on your PATH. Install it from https://cli.github.com and run `gh auth login`.",
    "git": "git is not on your PATH.",
}


@dataclass(frozen=True)
class Outcome:
    ok: bool
    message: str
    detail: str = ""


def dispatch_radar(*, runner: Callable = subprocess.run) -> Outcome:
    """Ask GitHub to start a Daily radar run.

    No inputs are passed: `hours` and `max_steps` carry defaults in the workflow
    itself, and sending them from here would mean two places to change when one
    of them moves.
    """
    return _run(
        ["gh", "workflow", "run", WORKFLOW],
        runner=runner,
        success="Radar dispatched. A pass takes about fifteen minutes.",
        failure="Could not dispatch the workflow.",
    )


def pull(*, runner: Callable = subprocess.run) -> Outcome:
    """Fast-forward the repo to whatever the workflow has committed.

    Pulls and nothing else — importing is the caller's next step.

    `--ff-only` is deliberate. The bot's commit always lands on top of main, so
    a fast-forward is all that is ever needed; when it is not enough it is
    because there is divergent local work, and a button that silently merged
    that would be the wrong tool for the moment.
    """
    return _run(
        ["git", "pull", "--ff-only"],
        runner=runner,
        success="Repo up to date.",
        failure="Could not fast-forward. Sort the local state out by hand.",
    )


def _run(argv: list[str], *, runner: Callable, success: str, failure: str) -> Outcome:
    try:
        completed = runner(
            argv,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        # Streamlit would otherwise render a traceback, which does not tell
        # anyone to go and install something.
        return Outcome(False, MISSING[argv[0]])
    except subprocess.TimeoutExpired:
        return Outcome(False, f"{argv[0]} took more than {TIMEOUT_SECONDS}s and was given up on.")

    if completed.returncode != 0:
        # Whatever the tool said is the only thing that explains the failure,
        # so it is carried through rather than replaced.
        return Outcome(False, failure, (completed.stderr or "").strip())

    return Outcome(True, success)
