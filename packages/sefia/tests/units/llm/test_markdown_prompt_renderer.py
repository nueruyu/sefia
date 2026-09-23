import json

import pytest

from sefia.llm import (
    InferencePrompt,
    MarkdownPromptRenderer,
    StructuredData,
)
from sefia.llm._text import markdown_fence
from sefia.llm.json_schema import JsonValue
from sefia.testing import make_function_info


def _renderer() -> MarkdownPromptRenderer:
    return MarkdownPromptRenderer()


def _task_content(arguments: StructuredData) -> str:
    return _renderer().render(
        InferencePrompt(
            function=make_function_info(),
            arguments=arguments,
            tools=(),
        )
    )


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
    content = _task_content(StructuredData.from_object({}))

    assert content.startswith("# Task\n\ninstructions")
    assert "## Response" not in content


def test_render_invocation_explains_when_there_are_no_direct_arguments():
    content = _task_content(StructuredData.from_object({}))

    assert "## Task arguments\n\nNone." in content


def test_render_renders_json_in_markdown():
    arguments: JsonValue = {
        "text": "日本語\nwith <markup> & symbols",
        "nested": {"enabled": True, "values": [1, None]},
    }

    prompt = _task_content(StructuredData.from_json(arguments))

    assert _json_content(prompt) == arguments
    assert "日本語" in prompt


def test_render_uses_a_fence_longer_than_content():
    prompt = _task_content(StructuredData.from_json({"source": "before ``` after"}))

    arguments = prompt.split("## Task arguments\n\n", 1)[1]
    assert arguments.splitlines()[0] == "````json"
    assert _json_content(prompt) == {"source": "before ``` after"}
