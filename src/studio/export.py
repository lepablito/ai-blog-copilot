"""Turn a finished draft into a Markdown file the blog will accept.

The front-matter contract belongs to the portfolio, not to us: every post is
validated with zod at build time (`src/lib/schema.ts`), and `relatedProject` is
a typed `reference()` — a slug naming no project fails `astro build` rather
than warning. So this module errs towards emitting less.

It writes to `output/posts/` and nothing else. Copying a post into the
portfolio and committing it is Pablo's decision, made once per post.
"""

import json
import re
import unicodedata
from datetime import date
from pathlib import Path

# Astro's content glob excludes files starting with "_", so templates are not
# real entries and must never be offered as a related project.
TEMPLATE_PREFIX = "_"

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    """ASCII kebab-case. This becomes both the filename and the URL.

    Accents are folded rather than dropped: "Diseño" should read "diseno", not
    "dise-o".
    """
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    slug = _NON_SLUG.sub("-", folded.lower()).strip("-")
    # A title made entirely of punctuation would otherwise write to ".md".
    return slug or "untitled"


def known_projects(portfolio_path: Path | str | None) -> list[str]:
    """Project slugs that really exist, sorted. Empty if not configured."""
    if portfolio_path is None:
        return []

    projects = Path(portfolio_path) / "content" / "projects"
    if not projects.is_dir():
        return []

    return sorted(
        entry.stem for entry in projects.glob("*.md") if not entry.name.startswith(TEMPLATE_PREFIX)
    )


def assemble(sections: list[tuple[str, str]]) -> str:
    """Sections into one Markdown body, each under its own `##` heading.

    Sections with no text yet are left out. Drafting happens one section at a
    time, so a half-written post is the normal state of things, and exporting
    a bare `##` with nothing under it helps nobody.
    """
    written = [(heading, text.strip()) for heading, text in sections if text.strip()]
    return "".join(f"## {heading}\n\n{text}\n\n" for heading, text in written).rstrip("\n") + (
        "\n" if written else ""
    )


def render_post(
    *,
    title: str,
    description: str,
    tags: list[str],
    body: str,
    today: date | None = None,
    related_project: str | None = None,
    portfolio_path: Path | str | None = None,
) -> str:
    """The full `.md`: YAML front-matter followed by the body."""
    fields = [
        f"title: {_yaml(title)}",
        f"description: {_yaml(description)}",
        f"date: {_yaml((today or date.today()).isoformat())}",
        f"tags: [{', '.join(_yaml(t.strip().lower()) for t in tags if t.strip())}]",
    ]

    if related_project and related_project in known_projects(portfolio_path):
        fields.append(f"relatedProject: {_yaml(related_project)}")

    # Always a draft. This tool proposes; publishing stays a decision Pablo
    # makes in his own repo, deliberately, once per post.
    fields.append("draft: true")

    return "---\n" + "\n".join(fields) + "\n---\n\n" + body


def write_post(
    output_dir: Path | str,
    *,
    overwrite: bool = False,
    **fields,
) -> Path:
    """Write the post to `<output_dir>/<slug>.md` and return the path.

    Refuses to overwrite unless asked: two posts on one topic in one day is a
    plausible mistake, and losing the first one to it is not recoverable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The title reaches the filename, and titles come from a model that read
    # untrusted pages. `slugify` already strips separators, but the containing
    # directory is asserted rather than assumed.
    path = output_dir / f"{slugify(fields['title'])}.md"
    if path.parent.resolve() != output_dir.resolve():
        raise ValueError(f"refusing to write outside {output_dir}")

    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists — pass overwrite=True to replace it")

    path.write_text(render_post(**fields), encoding="utf-8")
    return path


def _yaml(value: str) -> str:
    """Quote a scalar for YAML.

    `json.dumps` is exactly right here and not a shortcut: YAML 1.2 is a
    superset of JSON, so a JSON string is a valid YAML double-quoted scalar
    with the escaping already done. Titles contain colons and quotes often
    enough that hand-rolling this would eventually break someone's build.
    """
    return json.dumps(value, ensure_ascii=False)
