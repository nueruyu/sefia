import pytest

from sefia.llm._prompted_response import (
    PromptedJsonStreamExtractor,
    extract_prompted_json,
)


def test_extracts_raw_json() -> None:
    assert extract_prompted_json('  {"value":1}\n') == '{"value":1}'


def test_extracts_fenced_json_surrounded_by_prose() -> None:
    assert (
        extract_prompted_json(
            "Decision with {irrelevant} text:\n"
            '```json\n{"decision":"result","result":1}\n```\nDone.'
        )
        == '{"decision":"result","result":1}'
    )


def test_extracts_the_first_fence_without_validating_it() -> None:
    assert (
        extract_prompted_json(
            '```json\nnot json\n```\n```json\n{"decision":"result"}\n```'
        )
        == "not json"
    )


def test_extracts_a_fence_with_crlf_line_endings() -> None:
    assert extract_prompted_json('```json\r\n{"value":1}\r\n```') == '{"value":1}'


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"value":1}\nnot-a-closing-fence',
        '<function_calls>\n{"decision":"tool_calls"}\n</function_calls>',
        'first {"decision":"tool_calls"} then {"decision":"result"}',
        "There are 2. Result: 42",
    ],
)
def test_rejects_responses_without_raw_or_fenced_json(text: str) -> None:
    with pytest.raises(ValueError, match="contains no JSON"):
        extract_prompted_json(text)


@pytest.mark.parametrize(
    "content",
    [
        '{"decision":"result","result":1}',
        (
            "Decision with {irrelevant} text:\n"
            '```json\n{"decision":"result","result":1}\n```\nDone.'
        ),
    ],
)
def test_stream_extractor_returns_only_json_fragments(content: str) -> None:
    extractor = PromptedJsonStreamExtractor()

    extracted = "".join(extractor.feed(character) for character in content)

    assert extracted.strip() == '{"decision":"result","result":1}'


def test_stream_extractor_returns_json_before_the_fence_closes() -> None:
    extractor = PromptedJsonStreamExtractor()

    assert extractor.feed("Prose.\n```json\n") == ""
    assert extractor.feed('{"decision":"result"') == '{"decision":"result"'
