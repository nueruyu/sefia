import json
import re
from typing import Any, Callable, cast

from typing_extensions import final, override

from ._prompt_formatter import PromptFormatter

JsonDefault = Callable[[Any], Any]


@final
class MarkdownPromptFormatter(PromptFormatter):
    """Formats inference arguments as JSON in a Markdown code block."""

    def __init__(self, json_default: JsonDefault):
        self._json_default = json_default

    @override
    def format_arguments(self, arguments: dict[str, Any]) -> str:
        content = json.dumps(
            self._normalize(arguments),
            ensure_ascii=False,
            indent=2,
        )
        longest_run = max(
            (len(match.group()) for match in re.finditer(r"`+", content)),
            default=0,
        )
        fence = "`" * max(3, longest_run + 1)
        return f"## Task arguments\n\n{fence}json\n{content}\n{fence}"

    def _normalize(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            normalized: dict[str, Any] = {}
            for key, item in cast(dict[Any, Any], value).items():
                normalized_key = self._normalize_key(key)
                if normalized_key in normalized:
                    raise ValueError(
                        "Prompt argument mapping contains keys that normalize to "
                        f"the same JSON key: {normalized_key!r}"
                    )
                normalized[normalized_key] = self._normalize(item)
            return normalized
        if isinstance(value, (list, tuple)):
            sequence = cast(list[Any] | tuple[Any, ...], value)
            return [self._normalize(item) for item in sequence]
        try:
            normalized = self._json_default(value)
        except TypeError:
            return str(value)
        return str(value) if normalized is value else self._normalize(normalized)

    def _normalize_key(self, key: Any) -> str:
        normalized = self._normalize(key)
        if normalized is None:
            return "null"
        if normalized is True:
            return "true"
        if normalized is False:
            return "false"
        return normalized if isinstance(normalized, str) else str(normalized)
