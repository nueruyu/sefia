import json

from sefia.llm._text import compact_json, json_block


def test_compact_json_formats_normalized_data() -> None:
    rendered = compact_json({"text": "日本語", "items": [1, True, None]})

    assert rendered == '{"text":"日本語","items":[1,true,null]}'


def test_json_block_formats_normalized_data() -> None:
    rendered = json_block({"value": "before ``` after"})
    lines = rendered.splitlines()

    assert lines[0] == "````json"
    assert lines[-1] == "````"
    assert json.loads("\n".join(lines[1:-1])) == {"value": "before ``` after"}
