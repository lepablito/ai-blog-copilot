# Dismissing a topic from the Radar tab

**Date:** 2026-07-25
**Status:** approved, not yet implemented

## The problem

The radar proposes three to five topics a pass and keeps every one of them
forever. Some are duplicates of last week's, some are simply not worth writing.
There is no way to get them off the screen, so the list only grows and the good
suggestions get harder to see.

A plain `DELETE` does not work, and the reason is code shipped yesterday.
`import_json` re-inserts anything whose `(date, title)` is not already in the
topics table (`src/radar/store.py:210`), so a deleted row comes straight back
the next time "Fetch the latest topics" is clicked. Deleting has to mean
something that survives a re-import or it means nothing.

## Decisions

**A tombstone, not a delete.** The row stays and is marked. That is what makes
the dismissal survive re-import, and it keeps `radar.db` a complete record of
what the agent actually produced.

**Local to this machine.** `radar.db` is gitignored, so a dismissal is a
statement about this Studio, not about the archive. `data/topics.json` keeps
the topic. This matches the split the project already runs on: the committed
file is the durable copy, the database is the working one
(`Store.export_json`'s docstring says so directly).

**Identified by `(date, title)`.** The same identity `export_json` merges on and
`import_json` deduplicates on. A third notion of identity in a third place would
be a bug waiting to happen.

**Two-step confirmation.** A tombstone cannot be undone from the UI, so the
button asks once. Restoring from the UI is out of scope.

## Components

### Schema — `src/radar/store.py`

`topics` gains `dismissed_at TEXT`, NULL by default.

This needs a migration. `radar.db` already exists with data, and
`CREATE TABLE IF NOT EXISTS` does not add a column to a table that is already
there. `Store.__init__` runs `executescript(SCHEMA)` on every open
(`src/radar/store.py:60`); an idempotent `ALTER TABLE` goes immediately after
it, guarded by a `PRAGMA table_info(topics)` check. It fires once on an old
database and does nothing on every open after that.

Adding the column to `SCHEMA` as well keeps a freshly created database and a
migrated one identical.

### `Store.dismiss(date, title) -> int`

Sets `dismissed_at` to now for matching rows and returns how many it marked.
Dismissing something that is not there returns 0 rather than raising — the row
may have been dismissed already in another tab.

### `_select` gains `WHERE dismissed_at IS NULL`

One change at the single point every read passes through
(`src/radar/store.py:108`), so `recent_topics`, `recent_records` and
`export_records` all filter dismissed topics out with no further edits.

### Radar tab — `src/studio/views/radar.py`

Each card gets **Dismiss** beside the existing "Write this one". The first click
sets a flag in `st.session_state` keyed by the card; the redraw replaces the
button with "Sure?" and Yes / No.

`st.rerun()` **is** needed after dismissing, unlike the fetch button added
yesterday. The cards are drawn after `load_topics` has already run, so without a
rerun the dismissed card stays on screen until something else redraws the page.
The contrast with the neighbouring control invites someone to "fix" one to match
the other, so it gets a comment saying why they differ.

## The two consequences that matter

**The dismissal survives re-import, and `import_json` needs no change.** Its
duplicate check is a raw `SELECT date, title FROM topics`
(`src/radar/store.py:210`) that does not go through `_select`, so it sees
dismissed rows too, concludes the topics are already present, and skips them.

This falls out for free, and *because* it falls out for free it is fragile: a
later tidy-up that routes that query through `_select` would silently resurrect
every dismissed topic on the next fetch. It gets a dedicated test so that change
fails loudly instead.

**A dismissed topic already in the archive stays there.** `export_records` does
go through `_select`, so a dismissed topic stops being exported — but
`export_json` merges with what is already in the file and the merge preserves
it. `data/topics.json` keeps the topic.

The precise claim is "already in the archive". A topic found by a *local* run,
dismissed before any export, would never reach `data/topics.json` at all. That
is defensible — you judged it not worth keeping — but it is a real difference
from the case above and worth stating rather than glossing.

Today it cannot happen: nothing in the Studio calls `export_json`, and the only
caller is `radar.run --export` on a runner, where no topic has ever been
dismissed. The gap opens the day an export button appears in the Studio, and
whoever adds it should decide then whether a dismissal should suppress a topic
from the archive or only from the view.

## Testing

- a dismissed topic disappears from `recent_records`
- and stays gone under an angle filter, which builds a different WHERE clause
- **a dismissed topic does not come back when `topics.json` is re-imported** —
  the test guarding the property above
- `export_json` does not drop a dismissed topic from an existing file
- dismissing a topic that does not exist returns 0 and raises nothing
- dismissing one topic leaves its neighbours alone
- **migration**: a database created with the pre-column schema opens cleanly,
  its existing rows read back, and a topic in it can be dismissed

The migration test builds the old schema by hand rather than checking out an old
revision, so it keeps working when `SCHEMA` moves on.

## Out of scope

Restoring a dismissed topic from the UI, ruled out by choosing confirmation over
a visible-dismissed filter. It is one query and one button away if it turns out
to be wanted; nothing here would need reworking.

Propagating a dismissal to other machines. That would mean writing to
`data/topics.json` and committing from inside the Studio, and `export_json` is
written specifically never to lose topics from the archive.
