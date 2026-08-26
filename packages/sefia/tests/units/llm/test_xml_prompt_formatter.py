import xml.etree.ElementTree as ET
from dataclasses import dataclass

from sefia.llm import XmlPromptFormatter
from sefia.pydantic._json_utils import pydantic_json_default


@dataclass
class _CustomValue:
    value: str


def _formatter() -> XmlPromptFormatter:
    return XmlPromptFormatter(json_default=pydantic_json_default)


def test_format_arguments_escapes_xml_special_characters_in_strings() -> None:
    value = '<tag enabled="true">A & B</tag>'
    prompt = _formatter().format_arguments(arguments={"value": value})
    assert "&lt;tag" in prompt
    assert "&amp;" in prompt
    assert "<tag" not in prompt
    element = ET.fromstring(prompt).find("./argument[@name='value']/string")
    assert element is not None
    assert element.text == value


def test_format_arguments_falls_back_to_string_when_json_default_rejects_value() -> (
    None
):
    def json_default(_value: object) -> object:
        raise TypeError

    prompt = XmlPromptFormatter(json_default=json_default).format_arguments(
        arguments={"value": _CustomValue("fallback")}
    )
    element = ET.fromstring(prompt).find("./argument[@name='value']/string")
    assert element is not None
    assert element.text == "_CustomValue(value='fallback')"
