import json
import re
from collections.abc import Callable
from typing import cast

from typing_extensions import final, override

from ..inference import ToolCallsDecision
from ._prompt_renderer import (
    DecisionPrompt,
    DecisionResponseInstructions,
    PromptRenderer,
)
from .json_schema import JsonValue
from .step_decision import DecisionSpec, StepTool

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
    def render(self, prompt: DecisionPrompt) -> str:
        sections = [f"# Task\n\n{prompt.function.instructions}"]
        sections.append(self._render_arguments(prompt))
        if prompt.decision.tools:
            sections.append(self._render_tools(prompt.decision))
        if prompt.history:
            sections.append(self._render_history(prompt))
        sections.append(self._render_response(prompt.response))
        if prompt.rejected is not None:
            sections.append(self._render_rejection(prompt))
        return "\n\n".join(sections)

    def _render_arguments(self, prompt: DecisionPrompt) -> str:
        arguments = prompt.function.prompt_arguments
        if not arguments:
            return "## Task arguments\n\nNone."
        return "## Task arguments\n\n" + self._json_block(arguments)

    def _render_tools(self, decision: DecisionSpec) -> str:
        tools = "\n".join(self._render_tool(tool) for tool in decision.tools)
        return f"## Available tools\n\n{tools}"

    def _render_tool(self, tool: StepTool) -> str:
        description = f" — {tool.description}" if tool.description else ""
        schema = tool.arguments.to_dict()
        return (
            f"- `{tool.name}`{description}\n  Arguments: {self._compact_json(schema)}"
        )

    def _render_history(self, prompt: DecisionPrompt) -> str:
        records: list[JsonValue] = []
        for item in prompt.history:
            if isinstance(item, ToolCallsDecision):
                records.extend(
                    {
                        "tool_call": {
                            "id": call.id,
                            "name": call.name,
                            "arguments": self._normalize(call.arguments),
                        }
                    }
                    for call in item.calls
                )
            else:
                records.append(
                    {
                        "tool_result": {
                            "id": item.tool_call_id,
                            "result": self._normalize(item.result),
                        }
                    }
                )
        return "## Previous tool interactions\n\n" + self._json_block(records)

    def _render_response(self, response: DecisionResponseInstructions) -> str:
        forms: list[str] = []
        for form in response.forms:
            rendered = f"{form.label}:\n{self._code_block(form.example, 'json')}"
            if form.schema is not None:
                rendered += f"\nResult JSON Schema: {self._compact_json(form.schema)}"
            forms.append(rendered)
        content = [*forms, "\n".join(f"- {rule}" for rule in response.rules)]
        return "## Response\n\n" + "\n\n".join(part for part in content if part)

    def _render_rejection(self, prompt: DecisionPrompt) -> str:
        assert prompt.rejected is not None
        previous = (
            "The previous response was empty."
            if not prompt.rejected.content
            else "Previous response:\n" + self._text_block(prompt.rejected.content)
        )
        return (
            "## Correct the previous response\n\n"
            f"{previous}\n\nReason: {prompt.rejected.reason}\n\n"
            "Return a corrected response matching the Response section."
        )

    def _json_block(self, value: object) -> str:
        content = json.dumps(self._normalize(value), ensure_ascii=False, indent=2)
        return self._code_block(content, "json")

    @staticmethod
    def _code_block(content: str, language: str) -> str:
        fence = _markdown_fence(content)
        return f"{fence}{language}\n{content}\n{fence}"

    @staticmethod
    def _text_block(value: str) -> str:
        fence = _markdown_fence(value)
        return f"{fence}text\n{value}\n{fence}"

    def _compact_json(self, value: object) -> str:
        return json.dumps(
            self._normalize(value), ensure_ascii=False, separators=(",", ":")
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
