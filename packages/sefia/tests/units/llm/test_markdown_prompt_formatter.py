import json
from dataclasses import dataclass

from sefia.llm import MarkdownPromptFormatter
from sefia.pydantic._json_utils import pydantic_json_default


@dataclass
class _CustomValue:
    value: str


def _formatter() -> MarkdownPromptFormatter:
    return MarkdownPromptFormatter(json_default=pydantic_json_default)


def _json_content(prompt: str) -> object:
    lines = prompt.splitlines()
    assert lines[0] == "## Task arguments"
    assert lines[2].endswith("json")
    assert lines[-1] == lines[2][:-4]
    return json.loads("\n".join(lines[3:-1]))


def test_format_arguments_renders_json_in_markdown():
    arguments = {
        "text": "日本語\nwith <markup> & symbols",
        "nested": {"enabled": True, "values": [1, None]},
    }

    prompt = _formatter().format_arguments(arguments=arguments)

    assert _json_content(prompt) == arguments
    assert "日本語" in prompt


def test_format_arguments_uses_a_fence_longer_than_content():
    prompt = _formatter().format_arguments(arguments={"source": "before ``` after"})

    assert prompt.splitlines()[2] == "````json"
    assert _json_content(prompt) == {"source": "before ``` after"}


def test_format_arguments_uses_json_default():
    prompt = _formatter().format_arguments(
        arguments={"value": _CustomValue("serialized")}
    )

    assert _json_content(prompt) == {"value": {"value": "serialized"}}


def test_format_arguments_falls_back_to_string_when_json_default_rejects_value():
    def json_default(_value: object) -> object:
        raise TypeError

    prompt = MarkdownPromptFormatter(json_default=json_default).format_arguments(
        arguments={"value": _CustomValue("fallback")}
    )

    assert _json_content(prompt) == {"value": "_CustomValue(value='fallback')"}
