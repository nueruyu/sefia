import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pytest

from sefia.exceptions import InvalidInferenceResponseError
from sefia.inference import (
    FunctionInfo,
    ToolCallsDecision,
    ToolCallRequest,
    ToolCallResult,
)
from sefia.llm import MarkdownPromptRenderer
from sefia.llm._markdown_prompt_renderer import _markdown_fence
from sefia.llm.step_decision import StepDecisionSpec
from sefia.pydantic._json_utils import pydantic_json_default


@dataclass
class _CustomValue:
    value: str


def _renderer() -> MarkdownPromptRenderer:
    return MarkdownPromptRenderer(json_default=pydantic_json_default)


def _function_info(arguments: dict[str, Any] | None = None) -> FunctionInfo:
    return FunctionInfo(
        qualname="test",
        instructions="instructions",
        bound_arguments=arguments or {},
        type_hints={},
        return_type=str,
        args=(),
        kwargs={},
    )


def _decision_spec() -> StepDecisionSpec:
    return StepDecisionSpec.for_inference(
        name="StepDecision",
        output_type=str,
        tools=[],
    )


def _task_content(arguments: dict[str, Any]) -> str:
    messages = _renderer().render(
        _function_info(arguments),
        [],
        _decision_spec(),
    )
    assert messages[1].role == "user"
    return str(messages[1].content)


def _json_content(prompt: str) -> object:
    lines = prompt.splitlines()
    assert lines[0] == "## Task arguments"
    assert lines[2].endswith("json")
    assert lines[-1] == lines[2][:-4]
    return json.loads("\n".join(lines[3:-1]))


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", "```"),
        ("before ` after", "```"),
        ("before ``` after", "````"),
        ("before ```` after", "`````"),
    ],
)
def test_markdown_fence_is_longer_than_any_run_in_content(content: str, expected: str):
    assert _markdown_fence(content) == expected


def test_render_builds_the_complete_prompt():
    history = [
        ToolCallsDecision(
            calls=[
                ToolCallRequest(
                    id="1",
                    name="search",
                    arguments={"q": "日本語の検索クエリ"},
                )
            ]
        ),
        ToolCallResult(tool_call_id="1", result="見つかりました"),
    ]

    messages = _renderer().render(
        _function_info({"arg": "val"}),
        history,
        _decision_spec(),
    )

    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert str(messages[0].content).startswith("instructions")
    assert _json_content(str(messages[1].content)) == {"arg": "val"}
    assert json.loads(str(messages[2].content)) == {
        "decision": "tool_calls",
        "tool_calls": [
            {
                "id": "1",
                "name": "search",
                "arguments": {"q": "日本語の検索クエリ"},
            }
        ],
    }
    assert "\\u65e5" not in str(messages[2].content)
    assert json.loads(str(messages[3].content)) == {
        "tool_call_result": {
            "tool_call_id": "1",
            "result": "見つかりました",
        }
    }
    assert "\\u898b" not in str(messages[3].content)


def test_render_explains_when_there_are_no_direct_arguments():
    messages = _renderer().render(_function_info(), [], _decision_spec())

    assert "no direct function arguments" in str(messages[1].content)


def test_render_renders_json_in_markdown():
    arguments = {
        "text": "日本語\nwith <markup> & symbols",
        "nested": {"enabled": True, "values": [1, None]},
    }

    prompt = _task_content(arguments)

    assert _json_content(prompt) == arguments
    assert "日本語" in prompt


def test_render_uses_a_fence_longer_than_content():
    prompt = _task_content({"source": "before ``` after"})

    assert prompt.splitlines()[2] == "````json"
    assert _json_content(prompt) == {"source": "before ``` after"}


def test_render_uses_json_default():
    prompt = _task_content({"value": _CustomValue("serialized")})

    assert _json_content(prompt) == {"value": {"value": "serialized"}}


def test_render_normalizes_nested_mapping_keys():
    identifier = UUID("12345678-1234-5678-1234-567812345678")

    prompt = _task_content({"values_by_id": {identifier: "serialized"}})

    assert _json_content(prompt) == {"values_by_id": {str(identifier): "serialized"}}


def test_render_rejects_keys_that_normalize_to_the_same_json_key():
    identifier = UUID("12345678-1234-5678-1234-567812345678")

    with pytest.raises(ValueError, match="same JSON key"):
        _task_content({"values": {identifier: "first", str(identifier): "second"}})


def test_render_falls_back_to_string_when_json_default_rejects_value():
    def json_default(_value: object) -> object:
        raise TypeError

    renderer = MarkdownPromptRenderer(json_default=json_default)
    prompt = str(
        renderer.render(
            _function_info({"value": _CustomValue("fallback")}),
            [],
            _decision_spec(),
        )[1].content
    )

    assert _json_content(prompt) == {"value": "_CustomValue(value='fallback')"}


def test_render_repair_echoes_invalid_content_before_feedback():
    messages = _renderer().render_repair(
        InvalidInferenceResponseError("invalid schema", raw_content="invalid")
    )

    assert [message.role for message in messages] == ["assistant", "user"]
    assert messages[0].content == "invalid"
    assert "Error: invalid schema" in str(messages[1].content)


def test_render_repair_explains_an_empty_response():
    messages = _renderer().render_repair(
        InvalidInferenceResponseError("empty response")
    )

    assert [message.role for message in messages] == ["user"]
    assert "Your previous response was empty." in str(messages[0].content)
