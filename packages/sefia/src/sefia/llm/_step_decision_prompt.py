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
    sections = [_json_response_instruction(spec.mode)]
    if model.tools:
        sections.append(
            "History contains prior tool calls and results. Call only listed tools. "
            "Represent tool calls only with the JSON form above, never native tool-call "
            "syntax. Batch independent calls whose arguments are known; never guess "
            "arguments. Wait for a later step when arguments depend on a tool result; "
            "placeholders are invalid. A tool runs only after you return its JSON "
            "call; never continue to its result in the same response."
        )
        sections.append(
            "Available tools:\n" + "\n".join(_format_tool(tool) for tool in model.tools)
        )
    if model.result is not None:
        sections.append(
            "Final result JSON Schema:\n"
            + _compact_schema(model.result.schema.to_dict(), remove_titles=True)
        )
    return "\n\n" + "\n\n".join(sections)


def _json_response_instruction(mode: StepDecisionMode) -> str:
    header = "Return exactly one raw JSON object without prose or code fences."
    tool_shape = (
        "Tool call response:\n"
        '{"decision":"tool_calls","tool_calls":'
        '[{"name":"<tool name>","arguments":{}}]}'
    )
    result_shape = 'Final response:\n{"decision":"result","result":<final value>}'
    if mode is StepDecisionMode.TOOLS_REQUIRED:
        return f"{header} Call one or more tools; do not return a final result.\n\n{tool_shape}"
    if mode is StepDecisionMode.RESULT_ONLY:
        return (
            "Return only the non-null final JSON value without a decision envelope, "
            "prose, or code fences."
        )
    return (
        f"{header} Call tools when needed; return the result only when complete."
        f"\n\n{tool_shape}\n\n{result_shape}"
    )


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
