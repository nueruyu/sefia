import typing
import xml.dom.minidom as minidom
from typing import Any, Callable

from .._markers import AsRawText
from ._prompt_formatter import PromptFormatter

JsonDefault = Callable[[Any], Any]


class XmlPromptFormatter(PromptFormatter):
    """Formats inference arguments as well-formed XML for inclusion in LLM prompts."""

    def __init__(
        self,
        json_default: JsonDefault,
    ):
        self._json_default = json_default

    def format_arguments(
        self, arguments: dict[str, Any], type_hints: dict[str, Any]
    ) -> str:
        """Serialize prompt arguments as indented, well-formed XML."""
        document = minidom.Document()
        root = document.createElement("arguments")
        document.appendChild(root)

        for name, value in arguments.items():
            argument_element = document.createElement("argument")
            argument_element.setAttribute("name", name)
            type_hint = type_hints.get(name)
            self._append_value(document, argument_element, value, type_hint)
            root.appendChild(argument_element)

        return root.toprettyxml(indent="  ")

    def _append_cdata(
        self,
        document: minidom.Document,
        parent: minidom.Element,
        text: str,
    ) -> None:
        cdata_end = "]]" + ">"
        if cdata_end in text:
            # This escapes XML characters, but keeps prompt formatting simple for LLM input.
            parent.appendChild(document.createTextNode(text))
            return
        parent.appendChild(document.createCDATASection(text))

    def _append_value(
        self,
        document: minidom.Document,
        parent: minidom.Element,
        value: Any,
        type_hint: Any | None,
    ) -> None:
        if type_hint and typing.get_origin(type_hint) is typing.Annotated:
            metadata = typing.get_args(type_hint)[1:]
            if AsRawText in metadata and isinstance(value, str):
                element = document.createElement("string")
                self._append_cdata(document, element, value)
                parent.appendChild(element)
                return

        if value is None:
            parent.appendChild(document.createElement("null"))
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
            item_type_hint = self._get_mapping_value_type_hint(type_hint)
            for key, item in value.items():
                entry_element = document.createElement("entry")
                entry_element.setAttribute("key", str(key))
                self._append_value(document, entry_element, item, item_type_hint)
                object_element.appendChild(entry_element)
            parent.appendChild(object_element)
            return
        if isinstance(value, (list, tuple)):
            array_element = document.createElement("array")
            item_type_hint = self._get_sequence_item_type_hint(type_hint)
            for item in value:
                item_element = document.createElement("item")
                self._append_value(document, item_element, item, item_type_hint)
                array_element.appendChild(item_element)
            parent.appendChild(array_element)
            return

        try:
            normalized = self._json_default(value)
        except TypeError:
            normalized = value

        if value is normalized:
            element = document.createElement("string")
            element.appendChild(document.createTextNode(str(value)))
            parent.appendChild(element)
        else:
            self._append_value(document, parent, normalized, None)

    def _get_mapping_value_type_hint(self, type_hint: Any | None) -> Any | None:
        if type_hint is None:
            return None
        origin = typing.get_origin(type_hint)
        if origin is typing.Annotated:
            type_hint = typing.get_args(type_hint)[0]
            origin = typing.get_origin(type_hint)
        if origin not in (dict, typing.Dict):
            return None
        args = typing.get_args(type_hint)
        if len(args) != 2:
            return None
        return args[1]

    def _get_sequence_item_type_hint(self, type_hint: Any | None) -> Any | None:
        if type_hint is None:
            return None
        origin = typing.get_origin(type_hint)
        if origin is typing.Annotated:
            type_hint = typing.get_args(type_hint)[0]
            origin = typing.get_origin(type_hint)
        if origin not in (list, tuple, typing.List, typing.Tuple):
            return None
        args = typing.get_args(type_hint)
        if not args:
            return None
        return args[0]
