import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from uuid import UUID

import pytest

from sefia.inference import FunctionInfo
from sefia.llm import (
    InferencePrompt,
    MarkdownPromptRenderer,
)
from sefia.llm._text import markdown_fence
from sefia.llm.step_decision import DecisionSpec
from sefia.pydantic import PydanticModelBackend
from sefia.pydantic._json_utils import pydantic_json_default
from sefia.testing import make_function_info


@dataclass
class _CustomValue:
    value: str


def _renderer() -> MarkdownPromptRenderer:
    return MarkdownPromptRenderer(json_default=pydantic_json_default)


def _function_info(arguments: dict[str, Any] | None = None) -> FunctionInfo:
    return make_function_info(bound_arguments=arguments)


def _decision_spec() -> DecisionSpec:
    return DecisionSpec.for_inference(
        output_type=str,
        tools=[],
        result_format_factory=PydanticModelBackend(),
    )


def _prompt(
    function: FunctionInfo,
) -> InferencePrompt:
    return InferencePrompt(
        function=function,
        arguments=function.prompt_arguments,
        tools=_decision_spec().tools,
    )


def _task_content(arguments: dict[str, Any]) -> str:
    return _renderer().render(_prompt(_function_info(arguments)))


def _json_content(prompt: str) -> object:
    section = prompt.split("## Task arguments\n\n", 1)[1].split("\n\n##", 1)[0]
    lines = section.splitlines()
    assert lines[0].endswith("json")
    assert lines[-1] == lines[0][:-4]
    return json.loads("\n".join(lines[1:-1]))


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
    assert markdown_fence(content) == expected


def test_render_only_contains_inference_prompt():
    renderer = _renderer()
    content = renderer.render(_prompt(_function_info()))

    assert content.startswith("# Task\n\ninstructions")
    assert "## Response" not in content


def test_render_invocation_explains_when_there_are_no_direct_arguments():
    content = _renderer().render(_prompt(_function_info()))

    assert "## Task arguments\n\nNone." in content


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

    arguments = prompt.split("## Task arguments\n\n", 1)[1]
    assert arguments.splitlines()[0] == "````json"
    assert _json_content(prompt) == {"source": "before ``` after"}


def test_render_uses_json_default():
    prompt = _task_content({"value": _CustomValue("serialized")})

    assert _json_content(prompt) == {"value": {"value": "serialized"}}


def test_render_normalizes_nested_mapping_keys():
    identifier = UUID("12345678-1234-5678-1234-567812345678")

    prompt = _task_content({"values_by_id": {identifier: "serialized"}})

    assert _json_content(prompt) == {"values_by_id": {str(identifier): "serialized"}}


def test_render_normalizes_generic_mappings_as_json_objects():
    arguments = MappingProxyType({"knowledge": MappingProxyType({"foo": "bar"})})

    prompt = _renderer().render(
        InferencePrompt(function=_function_info(), arguments=arguments, tools=())
    )

    assert _json_content(prompt) == {"knowledge": {"foo": "bar"}}


def test_render_rejects_keys_that_normalize_to_the_same_json_key():
    identifier = UUID("12345678-1234-5678-1234-567812345678")

    with pytest.raises(ValueError, match="same JSON key"):
        _task_content({"values": {identifier: "first", str(identifier): "second"}})


def test_render_falls_back_to_string_when_json_default_rejects_value():
    def json_default(_value: object) -> object:
        raise TypeError

    renderer = MarkdownPromptRenderer(json_default=json_default)
    prompt = renderer.render(
        _prompt(_function_info({"value": _CustomValue("fallback")}))
    )

    assert _json_content(prompt) == {"value": "_CustomValue(value='fallback')"}
