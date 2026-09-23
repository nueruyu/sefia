import json

import pytest

from sefia.llm._markdown import json_block, markdown_fence


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", "```"),
        ("before ` after", "```"),
        ("before ``` after", "````"),
        ("before ```` after", "`````"),
    ],
)
def test_markdown_fence_is_longer_than_any_run_in_content(
    content: str, expected: str
) -> None:
    assert markdown_fence(content) == expected


def test_json_block_formats_normalized_data() -> None:
    rendered = json_block({"value": "before ``` after"})
    lines = rendered.splitlines()

    assert lines[0] == "````json"
    assert lines[-1] == "````"
    assert json.loads("\n".join(lines[1:-1])) == {"value": "before ``` after"}
