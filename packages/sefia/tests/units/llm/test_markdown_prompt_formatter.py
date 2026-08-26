import json
from dataclasses import dataclass
from uuid import UUID

import pytest

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


def test_format_arguments_normalizes_nested_mapping_keys():
    identifier = UUID("12345678-1234-5678-1234-567812345678")

    prompt = _formatter().format_arguments(
        arguments={"values_by_id": {identifier: "serialized"}}
    )

    assert _json_content(prompt) == {"values_by_id": {str(identifier): "serialized"}}


def test_format_arguments_rejects_keys_that_normalize_to_the_same_json_key():
    identifier = UUID("12345678-1234-5678-1234-567812345678")

    with pytest.raises(ValueError, match="same JSON key"):
        _formatter().format_arguments(
            arguments={"values": {identifier: "first", str(identifier): "second"}}
        )


def test_format_arguments_falls_back_to_string_when_json_default_rejects_value():
    def json_default(_value: object) -> object:
        raise TypeError

    prompt = MarkdownPromptFormatter(json_default=json_default).format_arguments(
        arguments={"value": _CustomValue("fallback")}
    )

    assert _json_content(prompt) == {"value": "_CustomValue(value='fallback')"}
