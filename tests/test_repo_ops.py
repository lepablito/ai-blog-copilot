"""The two buttons on the Radar tab, tested without running gh or git.

`repo_ops` takes its runner as an argument for exactly this reason. A test that
really dispatched a workflow would spend fifteen minutes of runner time and a
slice of the daily quota to assert a string.
"""

import subprocess

import pytest

from studio.repo_ops import dispatch_radar, pull


class FakeRunner:
    """Records the argv it was handed, then returns or raises to order."""

    def __init__(self, *, returncode: int = 0, stderr: str = "", raises: Exception | None = None):
        self.returncode = returncode
        self.stderr = stderr
        self.raises = raises
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(argv, self.returncode, stdout="", stderr=self.stderr)

    @property
    def argv(self) -> list[str]:
        ((argv, _),) = self.calls
        return argv

    @property
    def kwargs(self) -> dict:
        ((_, kwargs),) = self.calls
        return kwargs


# --- what gets run ---


def test_dispatch_asks_github_to_run_the_daily_workflow():
    runner = FakeRunner()

    dispatch_radar(runner=runner)

    assert runner.argv == ["gh", "workflow", "run", "daily-radar.yml"]


def test_dispatch_passes_no_inputs_so_the_workflow_defaults_apply():
    """`hours` and `max_steps` carry defaults in daily-radar.yml. Sending them
    from here would mean two places to change when one of them moves."""
    runner = FakeRunner()

    dispatch_radar(runner=runner)

    assert "-f" not in runner.argv


def test_pull_only_fast_forwards():
    """The bot's commit always lands on top of main, so a fast-forward is all
    that is ever needed. A button that quietly merged divergent local work
    would be the wrong tool for that moment."""
    runner = FakeRunner()

    pull(runner=runner)

    assert runner.argv == ["git", "pull", "--ff-only"]


@pytest.mark.parametrize("operation", [dispatch_radar, pull])
def test_neither_command_goes_through_a_shell(operation):
    runner = FakeRunner()

    operation(runner=runner)

    assert runner.kwargs.get("shell") is not True


# --- how it reports back ---


@pytest.mark.parametrize("operation", [dispatch_radar, pull])
def test_a_clean_exit_is_reported_as_success(operation):
    assert operation(runner=FakeRunner(returncode=0)).ok


@pytest.mark.parametrize("operation", [dispatch_radar, pull])
def test_a_failing_command_carries_its_stderr(operation):
    """Whatever gh or git said is the only thing that explains the failure, so
    it is shown rather than replaced with a message of our own."""
    runner = FakeRunner(returncode=1, stderr="fatal: not a git repository")

    outcome = operation(runner=runner)

    assert not outcome.ok
    assert "fatal: not a git repository" in outcome.detail


def test_a_missing_gh_says_so_by_name():
    """Streamlit would otherwise render a FileNotFoundError traceback, which
    does not tell you to install anything."""
    outcome = dispatch_radar(runner=FakeRunner(raises=FileNotFoundError()))

    assert not outcome.ok
    assert "gh" in outcome.message


def test_a_missing_git_says_so_by_name():
    outcome = pull(runner=FakeRunner(raises=FileNotFoundError()))

    assert not outcome.ok
    assert "git" in outcome.message


@pytest.mark.parametrize("operation", [dispatch_radar, pull])
def test_a_command_that_hangs_is_given_up_on(operation):
    """Streamlit runs this on the thread drawing the page. Without a timeout a
    stuck subprocess freezes the tab with no way back."""
    runner = FakeRunner(raises=subprocess.TimeoutExpired(cmd="x", timeout=1))

    outcome = operation(runner=runner)

    assert not outcome.ok
    assert runner.kwargs.get("timeout")
