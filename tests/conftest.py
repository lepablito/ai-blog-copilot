"""Suite-wide guards.

Nothing here sets up test data. The one fixture is a tripwire.
"""

from pathlib import Path

import pytest

# The database the Studio and the radar actually use, sitting in the repo root
# because `--db` and `DB_PATH` both default to a relative "radar.db".
LIVE_DB = Path(__file__).resolve().parents[1] / "radar.db"


def _fingerprint() -> tuple | None:
    if not LIVE_DB.exists():
        return None
    stat = LIVE_DB.stat()
    return (stat.st_size, stat.st_mtime_ns)


@pytest.fixture(autouse=True, scope="session")
def the_real_database_is_left_alone():
    """Fail the session if a test writes to the developer's radar.db.

    Three CLI tests used to call `main()` with no `--db`, which defaults to a
    relative "radar.db" — so every run of the suite quietly filed three topics
    titled "A topic" into the real database, where they showed up in the Radar
    tab alongside genuine ones. Nothing failed, so nobody noticed.

    A test that needs a database gets `tmp_path`. If this fires, that is what
    the offending test is missing.
    """
    before = _fingerprint()
    yield
    after = _fingerprint()

    if before != after:
        pytest.fail(
            f"a test wrote to {LIVE_DB}. Tests must use tmp_path: pass --db, "
            "or the db_path argument, a path under it."
        )
