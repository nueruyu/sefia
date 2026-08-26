import json
import re
from typing import Any, Callable

from typing_extensions import final, override

from ._prompt_formatter import PromptFormatter

JsonDefault = Callable[[Any], Any]


@final
class MarkdownPromptFormatter(PromptFormatter):
    """Formats inference arguments as JSON in a Markdown code block."""

    def __init__(self, json_default: JsonDefault):
        self._json_default = json_default

    @override
    def format_arguments(
        self, arguments: dict[str, Any], type_hints: dict[str, Any]
    ) -> str:
        del type_hints
        content = json.dumps(
            arguments,
            default=self._serialize,
            ensure_ascii=False,
            indent=2,
        )
        longest_run = max(
            (len(match.group()) for match in re.finditer(r"`+", content)),
            default=0,
        )
        fence = "`" * max(3, longest_run + 1)
        return f"## Task arguments\n\n{fence}json\n{content}\n{fence}"

    def _serialize(self, value: Any) -> Any:
        try:
            normalized = self._json_default(value)
        except TypeError:
            return str(value)
        return str(value) if normalized is value else normalized
