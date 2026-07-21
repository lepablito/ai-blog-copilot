import pytest

from radar.schema import InvalidTopics, Topic, parse_topics


def payload(**overrides):
    topic = {
        "title": "Prompt injection in agent tool loops",
        "summary": "Why tool observations are the weak point in ReAct agents.",
        "sources": ["https://example.com/a", "https://example.com/b"],
        "why_now": "Three separate write-ups landed this week.",
        "angle": "practical",
        "estimated_effort": "medium",
        "suggested_outline": ["The attack", "Why filters fail", "Structural defence"],
        "citations": ["https://example.com/a"],
    }
    return {**topic, **overrides}


def test_a_well_formed_payload_becomes_topics():
    topics = parse_topics({"topics": [payload()]})

    (topic,) = topics
    assert isinstance(topic, Topic)
    assert topic.title == "Prompt injection in agent tool loops"
    assert topic.angle == "practical"
    assert topic.sources == ["https://example.com/a", "https://example.com/b"]


def test_a_bare_list_is_accepted_too():
    """Models wrap the array about half the time. Both readings are unambiguous."""
    assert len(parse_topics([payload(), payload()])) == 2


@pytest.mark.parametrize(
    "field", ["title", "summary", "sources", "why_now", "angle", "suggested_outline"]
)
def test_missing_required_field_is_rejected(field):
    broken = payload()
    del broken[field]

    with pytest.raises(InvalidTopics, match=field):
        parse_topics({"topics": [broken]})


def test_error_message_names_the_offending_topic():
    """The message goes back to the model as a repair prompt — it has to be
    specific enough to act on."""
    with pytest.raises(InvalidTopics, match="topic 2"):
        parse_topics({"topics": [payload(), payload(angle="sideways")]})


def test_unknown_angle_is_rejected():
    with pytest.raises(InvalidTopics, match="angle"):
        parse_topics({"topics": [payload(angle="sideways")]})


def test_angle_is_case_insensitive():
    assert parse_topics({"topics": [payload(angle="Practical")]})[0].angle == "practical"


def test_empty_sources_is_rejected():
    """A topic nobody can trace back to a source is an assertion, not a finding."""
    with pytest.raises(InvalidTopics, match="sources"):
        parse_topics({"topics": [payload(sources=[])]})


def test_non_http_source_is_rejected():
    with pytest.raises(InvalidTopics, match="sources"):
        parse_topics({"topics": [payload(sources=["javascript:alert(1)"])]})


def test_empty_outline_is_rejected():
    with pytest.raises(InvalidTopics, match="suggested_outline"):
        parse_topics({"topics": [payload(suggested_outline=[])]})


def test_no_topics_at_all_is_rejected():
    with pytest.raises(InvalidTopics):
        parse_topics({"topics": []})


def test_a_scalar_payload_is_rejected():
    with pytest.raises(InvalidTopics):
        parse_topics("three topics, trust me")


def test_missing_optional_fields_get_sane_defaults():
    minimal = payload()
    del minimal["citations"]
    del minimal["estimated_effort"]

    (topic,) = parse_topics({"topics": [minimal]})

    assert topic.citations == []
    assert topic.estimated_effort == "unknown"


def test_a_topic_round_trips_through_a_dict():
    (topic,) = parse_topics({"topics": [payload()]})

    assert topic.as_dict()["angle"] == "practical"
    assert topic.as_dict()["sources"] == topic.sources
