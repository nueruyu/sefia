import json
import re
from collections.abc import Callable
from typing import cast

from typing_extensions import final, override

from ..exceptions import InvalidInferenceResponseError
from ..inference import FunctionInfo
from ._prompt_renderer import PromptRenderer
from ._step_decision_prompt import build_step_decision_prompt
from .json_schema import JsonValue
from .step_decision import StepDecisionSpec

JsonDefault = Callable[[object], object]


def _markdown_fence(content: str) -> str:
    longest_run = max(
        (len(match.group()) for match in re.finditer(r"`+", content)),
        default=0,
    )
    return "`" * max(3, longest_run + 1)


@final
class MarkdownPromptRenderer(PromptRenderer):
    """Renders inference prompts with Markdown and JSON content."""

    def __init__(self, json_default: JsonDefault):
        self._json_default = json_default

    @override
    def render_instructions(
        self,
        function_info: FunctionInfo,
        decision_spec: StepDecisionSpec,
    ) -> str:
        return function_info.instructions + build_step_decision_prompt(decision_spec)

    @override
    def render_invocation(self, function_info: FunctionInfo) -> str:
        arguments = function_info.prompt_arguments
        if not arguments:
            return (
                "This inference call has no direct function arguments. "
                "Follow the system instructions and use the conversation/tool "
                "history for any available context."
            )

        content = json.dumps(
            self._normalize(arguments),
            ensure_ascii=False,
            indent=2,
        )
        fence = _markdown_fence(content)
        return f"## Task arguments\n\n{fence}json\n{content}\n{fence}"

    @override
    def render_response_feedback(self, error: InvalidInferenceResponseError) -> str:
        content_note = (
            "" if error.raw_content else "Your previous response was empty.\n"
        )
        return (
            "Your previous response was invalid and could not be used as the "
            "required decision JSON.\n"
            f"Error: {error.detail}\n"
            f"{content_note}"
            "Respond again with exactly one valid raw JSON object matching the "
            "step-decision schema in the system instructions. Do not include prose, "
            "markdown, or code fences."
        )

    def _normalize(self, value: object) -> JsonValue:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            normalized: dict[str, JsonValue] = {}
            for key, item in cast(dict[object, object], value).items():
                normalized_key = self._normalize_key(key)
                if normalized_key in normalized:
                    raise ValueError(
                        "Prompt argument mapping contains keys that normalize to "
                        f"the same JSON key: {normalized_key!r}"
                    )
                normalized[normalized_key] = self._normalize(item)
            return normalized
        if isinstance(value, (list, tuple)):
            sequence = cast(list[object] | tuple[object, ...], value)
            return [self._normalize(item) for item in sequence]
        try:
            converted = self._json_default(value)
        except TypeError:
            return str(value)
        return str(value) if converted is value else self._normalize(converted)

    def _normalize_key(self, key: object) -> str:
        normalized = self._normalize(key)
        if normalized is None:
            return "null"
        if normalized is True:
            return "true"
        if normalized is False:
            return "false"
        return normalized if isinstance(normalized, str) else str(normalized)
