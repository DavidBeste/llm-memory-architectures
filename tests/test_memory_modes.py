import pytest

from letta_research_chat.cli import MEMORY_MODE_NAMES, memory_mode_name, parse_memory_mode


@pytest.mark.parametrize("mode_id, name", MEMORY_MODE_NAMES.items())
def test_canonical_memory_mode_names_round_trip(mode_id: int, name: str) -> None:
    assert parse_memory_mode(name) == mode_id
    assert memory_mode_name(mode_id) == name


@pytest.mark.parametrize("mode_id", MEMORY_MODE_NAMES)
def test_legacy_numeric_memory_modes_remain_supported(mode_id: int) -> None:
    assert parse_memory_mode(mode_id) == mode_id
    assert parse_memory_mode(str(mode_id)) == mode_id


def test_memory_mode_names_are_case_and_separator_friendly() -> None:
    assert parse_memory_mode("GRAPH_RERANK") == 6
    assert parse_memory_mode(" profile-json ") == 15
    assert parse_memory_mode("PROFILE_LOCOMO_RERANK") == 17
    assert parse_memory_mode("PROFILE_LOCOMO_CONTEXT_RERANK") == 18
    assert parse_memory_mode("PROFILE_LOCOMO_EVENTS_RERANK") == 19


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("agent", 1),
        ("rerank", 3),
        ("graph-raw", 4),
        ("memobase", 9),
        ("locomo", 16),
        ("locomo-rerank", 17),
        ("locomo-context-rerank", 18),
        ("locomo-events-rerank", 19),
    ],
)
def test_convenience_aliases(alias: str, expected: int) -> None:
    assert parse_memory_mode(alias) == expected


@pytest.mark.parametrize("value", ["20", "unknown", -1])
def test_unknown_memory_modes_have_actionable_errors(value: str | int) -> None:
    with pytest.raises(ValueError, match="memory mode"):
        parse_memory_mode(value)
