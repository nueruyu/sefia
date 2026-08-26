import json
from dataclasses import dataclass
from uuid import UUID

import pytest

from sefia.llm import MarkdownArgumentsRenderer
from sefia.llm._markdown_arguments_renderer import _markdown_fence
from sefia.pydantic._json_utils import pydantic_json_default


@dataclass
class _CustomValue:
    value: str


def _renderer() -> MarkdownArgumentsRenderer:
    return MarkdownArgumentsRenderer(json_default=pydantic_json_default)


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


def test_render_renders_json_in_markdown():
    arguments = {
        "text": "日本語\nwith <markup> & symbols",
        "nested": {"enabled": True, "values": [1, None]},
    }

    prompt = _renderer().render(arguments=arguments)

    assert _json_content(prompt) == arguments
    assert "日本語" in prompt


def test_render_uses_a_fence_longer_than_content():
    prompt = _renderer().render(arguments={"source": "before ``` after"})

    assert prompt.splitlines()[2] == "````json"
    assert _json_content(prompt) == {"source": "before ``` after"}


def test_render_uses_json_default():
    prompt = _renderer().render(arguments={"value": _CustomValue("serialized")})

    assert _json_content(prompt) == {"value": {"value": "serialized"}}


def test_render_normalizes_nested_mapping_keys():
    identifier = UUID("12345678-1234-5678-1234-567812345678")

    prompt = _renderer().render(arguments={"values_by_id": {identifier: "serialized"}})

    assert _json_content(prompt) == {"values_by_id": {str(identifier): "serialized"}}


def test_render_rejects_keys_that_normalize_to_the_same_json_key():
    identifier = UUID("12345678-1234-5678-1234-567812345678")

    with pytest.raises(ValueError, match="same JSON key"):
        _renderer().render(
            arguments={"values": {identifier: "first", str(identifier): "second"}}
        )


def test_render_falls_back_to_string_when_json_default_rejects_value():
    def json_default(_value: object) -> object:
        raise TypeError

    prompt = MarkdownArgumentsRenderer(json_default=json_default).render(
        arguments={"value": _CustomValue("fallback")}
    )

    assert _json_content(prompt) == {"value": "_CustomValue(value='fallback')"}
