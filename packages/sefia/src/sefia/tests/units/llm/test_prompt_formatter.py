import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from sefia.llm.prompt_formatter import PromptFormatter
from sefia.models import TextBlock
from sefia.pydantic.json_utils import pydantic_json_default


@dataclass
class _CustomValue:
    value: str


def _formatter() -> PromptFormatter:
    return PromptFormatter(json_default=pydantic_json_default)


def test_format_arguments_renders_text_blocks_in_cdata():
    source = 'if a < b and c > d:\n    print("A & B")\n'

    prompt = _formatter().format_arguments(
        {"file_contents": {"example.py": TextBlock(value=source)}}
    )

    root = ET.fromstring(prompt)
    text_block = root.find(
        "./argument[@name='file_contents']/object/entry[@key='example.py']/text_block"
    )
    assert text_block is not None
    assert text_block.attrib == {}
    assert text_block.text == f"\n{source}\n"
    assert "< b and c > d" in prompt
    assert "A & B" in prompt


def test_format_arguments_escapes_xml_special_characters_in_strings():
    value = '<tag enabled="true">A & B</tag>'

    prompt = _formatter().format_arguments({"value": value})

    assert '&lt;tag enabled="true"&gt;A &amp; B&lt;/tag&gt;' in prompt
    value_element = ET.fromstring(prompt).find("./argument[@name='value']/string")
    assert value_element is not None
    assert value_element.text == value


def test_format_arguments_handles_cdata_delimiters_in_text_blocks():
    source = "const marker = ']]>';\n"

    prompt = _formatter().format_arguments({"source": TextBlock(value=source)})

    text_block = ET.fromstring(prompt).find("./argument[@name='source']/text_block")
    assert text_block is not None
    assert text_block.text == f"\n{source}\n"


def test_format_json_preserves_unicode():
    payload = _formatter().format_json({"q": "日本語"})

    assert "\\u65e5" not in payload
    assert json.loads(payload) == {"q": "日本語"}
