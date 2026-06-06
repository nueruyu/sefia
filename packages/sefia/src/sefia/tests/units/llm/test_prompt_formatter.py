import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from sefia.llm.prompt_formatter import PromptFormatter
from sefia.pydantic.json_utils import pydantic_json_default


@dataclass
class _CustomValue:
    value: str


@dataclass
class _TextBlock:
    """A test-local version of the TextBlock concept."""

    value: str


def _formatter() -> PromptFormatter:
    return PromptFormatter(
        json_default=pydantic_json_default,
        text_block_selectors={_TextBlock: lambda tb: tb.value},
    )


def test_format_arguments_renders_selected_text_as_cdata_string():
    source = 'if a < b and c > d:\n    print("A & B")\n'

    prompt = _formatter().format_arguments(
        {"file_contents": {"example.py": _TextBlock(value=source)}}
    )

    root = ET.fromstring(prompt)
    string_element = root.find(
        "./argument[@name='file_contents']/object/entry[@key='example.py']/string"
    )
    assert string_element is not None
    assert string_element.attrib == {}
    assert string_element.text == source
    assert "<text_block" not in prompt
    assert "< b and c > d" in prompt
    assert "A & B" in prompt


def test_format_arguments_escapes_xml_special_characters_in_strings():
    value = '<tag enabled="true">A & B</tag>'

    prompt = _formatter().format_arguments({"value": value})

    assert '&lt;tag enabled="true"&gt;A &amp; B&lt;/tag&gt;' in prompt
    value_element = ET.fromstring(prompt).find("./argument[@name='value']/string")
    assert value_element is not None
    assert value_element.text == value


def test_format_arguments_handles_cdata_delimiters_in_selected_text():
    source = "const marker = ']]>';\n"

    prompt = _formatter().format_arguments({"source": _TextBlock(value=source)})

    string_element = ET.fromstring(prompt).find("./argument[@name='source']/string")
    assert string_element is not None
    assert string_element.text == source


def test_format_json_preserves_unicode():
    payload = _formatter().format_json({"q": "日本語"})

    assert "\\u65e5" not in payload
    assert json.loads(payload) == {"q": "日本語"}
