"""The Markdown exporter.

The contract is not ours: `Portfolio/src/lib/schema.ts` validates every post
with zod at build time, and `content.config.ts` upgrades `relatedProject` to a
typed `reference()` — a slug that names no project fails `astro build`. So the
tests here are about producing something that survives someone else's
validator, not about producing something that looks nice.
"""

from datetime import date

import pytest

from studio.export import (
    assemble,
    known_projects,
    list_posts,
    render_post,
    slugify,
    split_front_matter,
    write_post,
)


def test_split_front_matter_separates_the_yaml_from_the_body():
    text = '---\ntitle: "A post"\ndate: "2026-07-22"\n---\n\nThe body.\n'

    fields, body = split_front_matter(text)

    assert fields == {"title": "A post", "date": "2026-07-22"}
    assert body == "The body.\n"


def test_a_body_horizontal_rule_is_not_mistaken_for_the_fence():
    """`---` is also a Markdown horizontal rule. Only the first pair, at the
    very top, delimits front matter — a rule further down must stay in the
    body untouched."""
    text = '---\ntitle: "T"\n---\n\nOne.\n\n---\n\nTwo.\n'

    fields, body = split_front_matter(text)

    assert fields == {"title": "T"}
    assert body == "One.\n\n---\n\nTwo.\n"


def test_text_with_no_front_matter_is_all_body():
    fields, body = split_front_matter("Just prose, no fence.\n")

    assert fields == {}
    assert body == "Just prose, no fence.\n"


def test_front_matter_values_decode_json_scalars_and_arrays():
    text = '---\ntitle: "Quoted: colon"\ntags: ["a", "b"]\ndraft: true\n---\nBody\n'

    fields, _ = split_front_matter(text)

    assert fields["title"] == "Quoted: colon"
    assert fields["tags"] == ["a", "b"]
    assert fields["draft"] is True


def test_listing_posts_reads_title_and_date_newest_first(tmp_path):
    write_post(tmp_path, title="Older", description="d", tags=[], body="b", today=date(2026, 7, 20))
    write_post(tmp_path, title="Newer", description="d", tags=[], body="b", today=date(2026, 7, 22))

    posts = list_posts(tmp_path)

    assert [p["title"] for p in posts] == ["Newer", "Older"]
    assert posts[0]["date"] == "2026-07-22"
    assert posts[0]["slug"] == "newer"


def test_a_post_missing_its_title_falls_back_to_the_slug(tmp_path):
    (tmp_path / "hand-written.md").write_text("no front matter here\n", encoding="utf-8")

    [post] = list_posts(tmp_path)

    assert post["title"] == "hand-written"


def test_listing_a_directory_that_does_not_exist_is_empty(tmp_path):
    assert list_posts(tmp_path / "nowhere") == []


def test_assemble_puts_each_section_under_its_heading():
    body = assemble([("What it is", "A trick."), ("What it costs", "Memory.")])

    assert body == "## What it is\n\nA trick.\n\n## What it costs\n\nMemory.\n"


def test_assemble_skips_sections_nobody_has_written_yet():
    """The Studio drafts section by section, so a half-finished post is the
    normal case, not an error. Empty headings would export as dangling `##`."""
    body = assemble([("Written", "Text."), ("Not yet", "  ")])

    assert body == "## Written\n\nText.\n"


def test_slug_is_lowercase_kebab():
    assert slugify("Shipping LLM Systems") == "shipping-llm-systems"


def test_slug_strips_accents_rather_than_dropping_the_word():
    """The slug is the URL and the filename. 'Diseño' becoming 'dise-o' would
    be worse than useless."""
    assert slugify("Diseño de agentes") == "diseno-de-agentes"


def test_slug_collapses_punctuation_and_trims_the_edges():
    assert slugify("  RAG: what *actually* works?  ") == "rag-what-actually-works"


def test_a_title_with_nothing_slugifiable_still_produces_a_filename():
    """An empty filename would write to '.md' and silently overwrite it."""
    assert slugify("!!!") == "untitled"


def test_front_matter_carries_the_fields_the_blog_validates():
    post = render_post(
        title="Shipping LLM systems",
        description="What it involves.",
        tags=["LLM", "Evals"],
        body="Intro.\n",
        today=date(2026, 7, 22),
    )

    assert post.startswith("---\n")
    assert 'title: "Shipping LLM systems"' in post
    assert 'description: "What it involves."' in post
    assert 'date: "2026-07-22"' in post
    assert 'tags: ["llm", "evals"]' in post
    assert post.endswith("Intro.\n")


def test_export_is_always_a_draft():
    """Non-negotiable: this tool proposes, Pablo publishes. A post that landed
    in the portfolio already marked publishable would take the decision away
    from him."""
    post = render_post(title="T", description="D", tags=[], body="B", today=date(2026, 7, 22))

    assert "draft: true" in post
    assert "draft: false" not in post


def test_quotes_and_colons_in_a_title_do_not_break_the_yaml():
    """A colon in an unquoted YAML scalar starts a mapping. Titles routinely
    contain both colons and quotes, and the failure lands in someone else's
    build."""
    post = render_post(
        title='Attention: "all you need"?',
        description="A: B",
        tags=[],
        body="B",
        today=date(2026, 7, 22),
    )

    assert r'title: "Attention: \"all you need\"?"' in post


def test_related_project_is_included_when_the_slug_really_exists(tmp_path):
    projects = tmp_path / "content" / "projects"
    projects.mkdir(parents=True)
    (projects / "internal-rag-assistant.md").write_text("---\n---\n", encoding="utf-8")

    post = render_post(
        title="T",
        description="D",
        tags=[],
        body="B",
        today=date(2026, 7, 22),
        related_project="internal-rag-assistant",
        portfolio_path=tmp_path,
    )

    assert 'relatedProject: "internal-rag-assistant"' in post


def test_an_unknown_related_project_is_dropped_rather_than_emitted(tmp_path):
    """`relatedProject` is a typed reference() in the blog's content config, so
    a slug naming no project does not warn — it breaks `astro build`. Omitting
    it costs a link; emitting it costs Pablo a broken deploy."""
    (tmp_path / "content" / "projects").mkdir(parents=True)

    post = render_post(
        title="T",
        description="D",
        tags=[],
        body="B",
        today=date(2026, 7, 22),
        related_project="does-not-exist",
        portfolio_path=tmp_path,
    )

    assert "relatedProject" not in post


def test_related_project_is_dropped_when_the_portfolio_is_not_configured():
    """PORTFOLIO_PATH is optional. With no way to check the slug, the safe
    answer is to leave the field out."""
    post = render_post(
        title="T",
        description="D",
        tags=[],
        body="B",
        today=date(2026, 7, 22),
        related_project="internal-rag-assistant",
        portfolio_path=None,
    )

    assert "relatedProject" not in post


def test_template_files_are_not_offered_as_projects(tmp_path):
    """Astro's glob excludes `_*`. Offering `_TEMPLATE` in the dropdown would
    let someone pick a slug the build then rejects."""
    projects = tmp_path / "content" / "projects"
    projects.mkdir(parents=True)
    (projects / "_TEMPLATE.md").write_text("---\n---\n", encoding="utf-8")
    (projects / "real-one.md").write_text("---\n---\n", encoding="utf-8")

    assert known_projects(tmp_path) == ["real-one"]


def test_no_portfolio_configured_means_no_projects_and_no_crash(tmp_path):
    assert known_projects(None) == []
    assert known_projects(tmp_path / "nowhere") == []


def test_writing_a_post_names_the_file_after_the_slug(tmp_path):
    path = write_post(
        tmp_path,
        title="Shipping LLM systems",
        description="D",
        tags=[],
        body="Body.\n",
        today=date(2026, 7, 22),
    )

    assert path == tmp_path / "shipping-llm-systems.md"
    assert path.read_text(encoding="utf-8").endswith("Body.\n")


def test_writing_never_reaches_outside_the_output_directory(tmp_path):
    """The title reaches the slug, and the slug reaches the filesystem. A title
    of '../../etc/passwd' must not become a path."""
    path = write_post(
        tmp_path,
        title="../../escaped",
        description="D",
        tags=[],
        body="B",
        today=date(2026, 7, 22),
    )

    assert path.parent == tmp_path


def test_an_existing_post_is_not_silently_overwritten(tmp_path):
    kwargs = dict(title="Same title", description="D", tags=[], body="B", today=date(2026, 7, 22))
    write_post(tmp_path, **kwargs)

    with pytest.raises(FileExistsError):
        write_post(tmp_path, **kwargs)

    assert write_post(tmp_path, overwrite=True, **kwargs).exists()
