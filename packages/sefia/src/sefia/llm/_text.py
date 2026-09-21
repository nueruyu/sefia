import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import cast
from uuid import UUID

from .json_schema import JsonValue

JsonDefault = Callable[[object], object]


def markdown_fence(content: str) -> str:
    longest_run = max(
        (len(match.group()) for match in re.finditer(r"`+", content)),
        default=0,
    )
    return "`" * max(3, longest_run + 1)


def _code_block(content: str, language: str) -> str:
    fence = markdown_fence(content)
    return f"{fence}{language}\n{content}\n{fence}"


def text_block(value: str) -> str:
    return _code_block(value, "text")


def _generic_json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    raise TypeError


class TextFormatter:
    """Normalizes Python values for JSON text in prompts and transport messages."""

    def __init__(self, json_default: JsonDefault = _generic_json_default) -> None:
        self._json_default = json_default

    def json_block(self, value: object) -> str:
        content = json.dumps(self.normalize(value), ensure_ascii=False, indent=2)
        return _code_block(content, "json")

    def compact_json(self, value: object) -> str:
        return json.dumps(
            self.normalize(value), ensure_ascii=False, separators=(",", ":")
        )

    def normalize(self, value: object) -> JsonValue:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Mapping):
            normalized: dict[str, JsonValue] = {}
            for key, item in cast(Mapping[object, object], value).items():
                normalized_key = self._normalize_key(key)
                if normalized_key in normalized:
                    raise ValueError(
                        "Prompt argument mapping contains keys that normalize to "
                        f"the same JSON key: {normalized_key!r}"
                    )
                normalized[normalized_key] = self.normalize(item)
            return normalized
        if isinstance(value, (list, tuple)):
            sequence = cast(list[object] | tuple[object, ...], value)
            return [self.normalize(item) for item in sequence]
        try:
            converted = self._json_default(value)
        except TypeError:
            return str(value)
        return str(value) if converted is value else self.normalize(converted)

    def _normalize_key(self, key: object) -> str:
        normalized = self.normalize(key)
        if normalized is None:
            return "null"
        if normalized is True:
            return "true"
        if normalized is False:
            return "false"
        return normalized if isinstance(normalized, str) else str(normalized)
