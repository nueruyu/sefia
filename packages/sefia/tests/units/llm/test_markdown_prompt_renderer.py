import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pytest

from sefia.exceptions import InvalidInferenceResponseError
from sefia.inference import FunctionInfo
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
    return _renderer().render_invocation(_function_info(arguments))


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


def test_render_instructions_combines_function_and_decision_instructions():
    content = _renderer().render_instructions(_function_info(), _decision_spec())

    assert content.startswith("instructions")
    assert "Set `decision` to `result`" in content


def test_render_invocation_explains_when_there_are_no_direct_arguments():
    content = _renderer().render_invocation(_function_info())

    assert "no direct function arguments" in content


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
    prompt = renderer.render_invocation(
        _function_info({"value": _CustomValue("fallback")})
    )

    assert _json_content(prompt) == {"value": "_CustomValue(value='fallback')"}


def test_render_response_feedback_describes_the_error():
    feedback = _renderer().render_response_feedback(
        InvalidInferenceResponseError("invalid schema", raw_content="invalid")
    )

    assert "Error: invalid schema" in feedback
    assert "previous response was empty" not in feedback


def test_render_response_feedback_explains_an_empty_response():
    feedback = _renderer().render_response_feedback(
        InvalidInferenceResponseError("empty response")
    )

    assert "Your previous response was empty." in feedback
