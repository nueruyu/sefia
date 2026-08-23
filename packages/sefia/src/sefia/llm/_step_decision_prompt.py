import json

from .json_schema import JsonValue
from .step_decision import (
    JsonToolArguments,
    StepDecisionMode,
    StepDecisionModel,
    StepDecisionSpec,
    StepTool,
)


def build_step_decision_prompt(spec: StepDecisionSpec) -> str:
    if spec.mode is StepDecisionMode.TOOLS_REQUIRED:
        instruction = (
            "Call one or more tools by setting `decision` to `tool_calls` and "
            "populating `tool_calls`; do not return a final result."
        )
    elif spec.mode is StepDecisionMode.TOOLS_OR_RESULT:
        instruction = (
            "Set `decision` to `tool_calls` and populate `tool_calls` when tools "
            "are needed; otherwise, set `decision` to `result` and populate "
            "`result` only when the task is complete."
        )
    else:
        instruction = (
            "Set `decision` to `result` and populate `result` with the non-null "
            "task result."
        )
    if spec.tools:
        instruction += (
            " Treat all tool interactions in the conversation history as part of "
            "this decision protocol. Request new tool calls only through this "
            "response schema. Batch only independent tool calls with known arguments. "
            "Never guess arguments or use placeholders; defer calls that depend on "
            "another tool's result."
        )
    return f"\n\n{instruction}"


def build_json_decision_prompt(spec: StepDecisionSpec, model: StepDecisionModel) -> str:
    sections = [_json_response_format(spec.mode)]
    if model.tools:
        sections.append(_TOOL_RULES)
        sections.append(
            "## Tools\n" + "\n".join(_format_tool(tool) for tool in model.tools)
        )
    if model.result is not None:
        sections.append(
            "## Result schema\n"
            + _compact_schema(model.result.schema.to_dict(), remove_titles=True)
        )
    return "\n\n" + "\n\n".join(sections)


def _json_response_format(mode: StepDecisionMode) -> str:
    header = "## Response\nReturn JSON only. No prose, code fences, or XML."
    tool_shape = (
        "Tool call:\n"
        '{"decision":"tool_calls","tool_calls":'
        '[{"name":"<tool name>","arguments":{}}]}'
    )
    result_shape = 'Result:\n{"decision":"result","result":<final value>}'
    if mode is StepDecisionMode.TOOLS_REQUIRED:
        return f"{header}\n\n{tool_shape}"
    if mode is StepDecisionMode.RESULT_ONLY:
        return f"{header}\nReturn the non-null final value without an envelope."
    return f"{header}\n\n{tool_shape}\n\n{result_shape}"


_TOOL_RULES = """## Rules
- Use exact tool names from `Tools`.
- History contains previous calls and results.
- Use only the JSON forms above, never native tool-call syntax.
- Batch only independent calls with known arguments.
- Wait for dependent results; never guess or use placeholders.
- A call runs after your response. Do not return its result in the same response."""


def _format_tool(tool: StepTool) -> str:
    description = f" — {tool.description}" if tool.description else ""
    return f"- `{tool.name}`{description}\n  arguments JSON Schema: " + _compact_schema(
        tool.arguments.json_schema.to_dict(),
        remove_titles=not isinstance(tool.arguments, JsonToolArguments),
    )


def _compact_json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _compact_schema(value: JsonValue, *, remove_titles: bool) -> str:
    return _compact_json(_without_titles(value) if remove_titles else value)


def _without_titles(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: _without_titles(item) for key, item in value.items() if key != "title"
        }
    if isinstance(value, list):
        return [_without_titles(item) for item in value]
    return value
