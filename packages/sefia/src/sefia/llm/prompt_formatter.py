import json
import xml.dom.minidom as minidom
from typing import Any, Callable

from ..models import TextBlock

JsonDefault = Callable[[Any], Any]


class PromptFormatter:
    """Formats inference arguments for inclusion in LLM prompts."""

    def __init__(self, json_default: JsonDefault):
        self._json_default = json_default

    def format_arguments(self, arguments: dict[str, Any]) -> str:
        """Serialize prompt arguments as indented, well-formed XML."""
        document = minidom.Document()
        root = document.createElement("arguments")
        document.appendChild(root)

        for name, value in arguments.items():
            argument_element = document.createElement("argument")
            argument_element.setAttribute("name", name)
            self._append_value(document, argument_element, value)
            root.appendChild(argument_element)

        return root.toprettyxml(indent="  ")

    def _append_cdata(
        self,
        document: minidom.Document,
        parent: minidom.Element,
        text: str,
    ) -> None:
        if "]]>" in text:
            parent.appendChild(document.createTextNode(text))
            return
        parent.appendChild(document.createCDATASection(text))

    def _append_value(
        self,
        document: minidom.Document,
        parent: minidom.Element,
        value: Any,
    ) -> None:
        if value is None:
            parent.appendChild(document.createElement("null"))
            return
        if isinstance(value, TextBlock):
            text_block_element = document.createElement("text_block")
            self._append_cdata(document, text_block_element, f"\n{value.value}\n")
            parent.appendChild(text_block_element)
            return
        if isinstance(value, bool):
            element = document.createElement("boolean")
            element.appendChild(document.createTextNode(str(value).lower()))
            parent.appendChild(element)
            return
        if isinstance(value, int):
            element = document.createElement("integer")
            element.appendChild(document.createTextNode(str(value)))
            parent.appendChild(element)
            return
        if isinstance(value, float):
            element = document.createElement("number")
            element.appendChild(document.createTextNode(str(value)))
            parent.appendChild(element)
            return
        if isinstance(value, str):
            element = document.createElement("string")
            element.appendChild(document.createTextNode(value))
            parent.appendChild(element)
            return
        if isinstance(value, dict):
            object_element = document.createElement("object")
            for key, item in value.items():
                entry_element = document.createElement("entry")
                entry_element.setAttribute("key", str(key))
                self._append_value(document, entry_element, item)
                object_element.appendChild(entry_element)
            parent.appendChild(object_element)
            return
        if isinstance(value, (list, tuple)):
            array_element = document.createElement("array")
            for item in value:
                item_element = document.createElement("item")
                self._append_value(document, item_element, item)
                array_element.appendChild(item_element)
            parent.appendChild(array_element)
            return

        try:
            normalized = self._json_default(value)
        except TypeError:
            normalized = str(value)
        if normalized is not value:
            self._append_value(document, parent, normalized)
            return

        element = document.createElement("string")
        element.appendChild(document.createTextNode(str(value)))
        parent.appendChild(element)

    def format_json(self, value: Any) -> str:
        return json.dumps(
            value,
            default=self._json_default,
            ensure_ascii=False,
        )
