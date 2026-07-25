# Studio: launch the radar and bring its topics home

**Date:** 2026-07-25
**Status:** approved, not yet implemented

## The problem

The Radar tab reads `radar.db` and nothing else (`src/studio/views/radar.py:24`,
via `studio.radar_data.load_topics`). That file is gitignored (`.gitignore:10`)
and on a workflow run it exists only as a GitHub Actions cache.

What the nightly workflow actually commits is `data/topics.json`, and no module
under `src/studio/` reads it. The only reference to that path anywhere in `src/`
is the `--export` flag of `radar/run.py:100`.

So today the loop is broken in both directions. Starting a run means leaving the
Studio for the GitHub UI or a terminal, and the topics a run produces never
reach the tab that exists to show them — a `git pull` brings the JSON down and
the Radar tab keeps displaying exactly what it displayed before.

Two buttons close the loop: one to start a run, one to bring its results back.

## Decisions

**Import into `radar.db` rather than teaching the view to read JSON.** The Radar
tab keeps one source of truth, which is the property that lets
`tests/test_radar_view.py` test the whole tab without a running server. A view
merging two sources would need a rule for which copy wins when a topic appears
in both, and that rule would have to be right forever.

**`gh` CLI over a personal access token.** It reuses a session that is already
authenticated on this machine and adds no secret to `.env` — nothing to rotate
and nothing that can reach a commit. The cost is a dependency on a binary that
`pyproject.toml` does not declare, which is acceptable because the Studio is a
local tool started from `run.bat`, and which the error handling makes explicit
rather than mysterious.

**Fire and forget.** A pass takes about fifteen minutes. Polling would block the
Streamlit script or need threads and session state, to save a click on a button
that has to exist anyway.

## Components

### `src/studio/repo_ops.py` (new)

No `import streamlit`. Two functions wrapping external processes, following the
project's existing split: logic in plain modules, Streamlit only draws.

```python
@dataclass(frozen=True)
class Outcome:
    ok: bool
    message: str   # one line, for st.success / st.error
    detail: str    # stderr, shown in an expander when it is not empty

def dispatch_radar(*, runner=subprocess.run) -> Outcome
def pull(*, runner=subprocess.run) -> Outcome
```

`runner` is injected the same way the radar injects its fetcher. Tests pass a
double that records the argv it received and returns a chosen exit code; neither
`gh` nor `git` is ever really executed.

Both calls use a fixed argument list, never `shell=True`, and interpolate
nothing from the UI or from a model.

- `dispatch_radar` runs `gh workflow run daily-radar.yml`. The workflow's
  `hours` and `max_steps` inputs carry defaults, so none are passed.
- `pull` runs `git pull --ff-only`. It pulls and nothing else — importing is the
  view's next step, not something `pull` does on the way out. Keeping the two
  apart is what lets each be tested against a fake runner without a database.

Neither function sets a working directory. Both inherit the Studio's, which is
the repo root: `run.bat` does `pushd "%~dp0"` before launching Streamlit. A
Studio started from somewhere else is already broken in more basic ways —
`DB_PATH` in `app.py:25` is the relative `radar.db`.

### `Store.import_json()` (new, `src/radar/store.py`)

The mirror of `export_json`, keyed on `(date, title)` — the same identity that
`export_json` already merges on (`src/radar/store.py:158`). Returns the number
of topics inserted.

`topics.run_id` is `NOT NULL REFERENCES runs(id)`, so imported topics cannot
stand alone. The import opens one row in `runs` with
`goal="imported from data/topics.json"` and `status="imported"` and hangs the
new topics off it. That keeps an import distinguishable from a real pass in the
runs table, which is honest: it was not a pass.

A run row is created only when there is something to insert. An import that
finds nothing new leaves the database untouched.

### Radar tab (`src/studio/views/radar.py`)

Two buttons in two columns above the existing filters.

The `st.info` shown when there are no topics (`src/studio/views/radar.py:27`)
currently says "dispatch the Daily radar workflow". It gets updated to point at
the button instead.

## Flow

```
[Launch the radar]  → gh workflow run daily-radar.yml
                    → "Launched" + link to the run. Done.

[Fetch the topics]  → git pull --ff-only
                    → Store.import_json("data/topics.json")
                    → "3 new topics" + st.rerun()
```

The view orchestrates those two steps: it calls `pull()`, and only on
`Outcome.ok` does it call `import_json`. A failed pull reports the failure and
stops there — importing the stale file that is already on disk would print a
reassuring "0 new topics" over the top of a real error.

The `st.rerun()` is what repaints the list below. Without it the topics land in
the database and the tab keeps showing the previous query, so the button looks
like it did nothing.

## Error handling

Every failure is surfaced as it happened. Nothing is repaired automatically.

| Situation | What the user sees |
|---|---|
| `gh` not installed | `FileNotFoundError` → "gh is not on your PATH" plus the install link |
| `gh` not logged in | non-zero exit → stderr in an expander |
| `git pull` cannot fast-forward | `--ff-only` fails → stderr, and **no** merge or rebase is attempted |
| `data/topics.json` is corrupt | `_existing_topics` already raises `ValueError` with a usable message; it propagates |
| `data/topics.json` is absent | 0 topics imported, no error |

`--ff-only` is deliberate. The bot's commit always lands on top of `main`, so a
fast-forward is all that is ever needed. When it is not enough it is because
there is divergent local work, and a button that silently merged it would be
the wrong tool for that moment.

## Testing

`tests/test_repo_ops.py` (new):

- `dispatch_radar` invokes exactly `["gh", "workflow", "run", "daily-radar.yml"]`
- `pull` invokes exactly `["git", "pull", "--ff-only"]`
- exit 0 → `ok=True`; non-zero exit → `ok=False` carrying stderr in `detail`
- a runner raising `FileNotFoundError` → `ok=False` with the missing-binary message

Added to `tests/test_store.py`:

- importing a file inserts its topics and returns the count
- importing the same file twice inserts nothing the second time — the
  `(date, title)` key holds
- an absent file is a no-op returning 0, and creates no `runs` row
- a corrupt file raises `ValueError`
- an import that inserts nothing leaves the `runs` table untouched

## Out of scope

The workflow declares `hours` and `max_steps` inputs. They are not exposed in
the UI: the defaults are sensible and two more widgets cost permanent screen
space for something that will rarely be touched. They can be added later
without reworking any of this.

The Studio does not learn whether a dispatched run succeeded. That is what the
GitHub run link is for, and an import that brings back nothing says the same
thing a moment later.
