import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Annotated

from sefia import AsRawText
from sefia.llm.xml_prompt_formatter import XmlPromptFormatter
from sefia.pydantic.json_utils import pydantic_json_default


@dataclass
class _CustomValue:
    value: str


def _formatter() -> XmlPromptFormatter:
    return XmlPromptFormatter(
        json_default=pydantic_json_default,
    )


def test_format_arguments_renders_annotated_text_as_cdata_string():
    source = 'if a < b and c > d:\n    print("A & B")\n'
    RawCode = Annotated[str, AsRawText]

    prompt = _formatter().format_arguments(
        arguments={"file_contents": {"example.py": source}},
        type_hints={"file_contents": dict[str, RawCode]},
    )

    root = ET.fromstring(prompt)
    string_element = root.find(
        "./argument[@name='file_contents']/object/entry[@key='example.py']/string"
    )
    assert string_element is not None
    assert string_element.attrib == {}
    assert string_element.text == source
    assert "< b and c > d" in prompt
    assert "A & B" in prompt


def test_format_arguments_escapes_xml_special_characters_in_strings():
    value = '<tag enabled="true">A & B</tag>'

    prompt = _formatter().format_arguments(arguments={"value": value}, type_hints={})

    # Markup-significant characters are escaped (the exact escaping of quotes in
    # text nodes is left to the XML writer and varies across Python versions),
    # and no raw markup leaks into the output.
    assert "&lt;tag" in prompt
    assert "&amp;" in prompt
    assert "<tag" not in prompt
    # The value round-trips: parsing the prompt yields the original string back.
    value_element = ET.fromstring(prompt).find("./argument[@name='value']/string")
    assert value_element is not None
    assert value_element.text == value


def test_format_arguments_handles_cdata_delimiters_in_annotated_text():
    source = "const marker = ']]>';\n"
    RawCode = Annotated[str, AsRawText]

    prompt = _formatter().format_arguments(
        arguments={"source": source}, type_hints={"source": RawCode}
    )

    string_element = ET.fromstring(prompt).find("./argument[@name='source']/string")
    assert string_element is not None
    assert string_element.text == source


def test_format_arguments_falls_back_to_string_when_json_default_rejects_value():
    def json_default(_value):
        raise TypeError

    prompt = XmlPromptFormatter(json_default=json_default).format_arguments(
        arguments={"value": _CustomValue("fallback")}, type_hints={}
    )

    value_element = ET.fromstring(prompt).find("./argument[@name='value']/string")
    assert value_element is not None
    assert value_element.text == "_CustomValue(value='fallback')"
