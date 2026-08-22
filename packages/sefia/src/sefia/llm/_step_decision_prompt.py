import json

from .step_decision import StepDecisionMode, StepDecisionSpec

_TOOL_DEFINITIONS_HEADER = (
    "\n### Available Tools\n"
    "Here is a list of tools you can call. Use their `name` in the `tool_calls` field.\n"
)
_TOOL_CALLS_RESPONSE_FORMAT = (
    "To call tools, select the `tool_calls` decision and provide each tool's "
    "name and arguments according to the response schema."
)
_RESULT_RESPONSE_FORMAT = (
    "To complete the task, select the `result` decision and provide the result "
    "according to the response schema."
)


def build_step_decision_prompt(spec: StepDecisionSpec) -> str:
    if spec.mode is StepDecisionMode.TOOLS_REQUIRED:
        instructions = (
            "Your task is to call tools. You MUST set `decision` to `tool_calls` "
            "and populate the `tool_calls` field. There is no `result` — "
            "you must never stop calling tools."
        )
        formats = _TOOL_CALLS_RESPONSE_FORMAT
    elif spec.mode is StepDecisionMode.TOOLS_OR_RESULT:
        instructions = (
            "Your task is to decide the next step. You have two options:\n"
            "1. Call one or more tools by setting `decision` to `tool_calls` "
            "and populating the `tool_calls` field.\n"
            "2. Complete the task by setting `decision` to `result` "
            "and populating the `result` field.\n\n"
            "Use `tool_calls` to gather more information, and use `result` "
            "only when you have enough information to complete the entire task."
        )
        formats = f"{_TOOL_CALLS_RESPONSE_FORMAT}\n{_RESULT_RESPONSE_FORMAT}"
    else:
        instructions = (
            "Your task is to provide a non-null result by setting `decision` "
            "to `result` and populating the `result` field. No tools are "
            "available. If the requested result is a collection and there are "
            "no results, return an empty collection instead of null."
        )
        formats = _RESULT_RESPONSE_FORMAT

    tools = ""
    if spec.tools:
        definitions = [tool.definition().to_dict() for tool in spec.tools]
        tools = (
            f"{_TOOL_DEFINITIONS_HEADER}"
            f"{json.dumps(definitions, indent=2, ensure_ascii=False)}\n"
        )
    return f"\n\n### Response Instructions\n{instructions}\n{tools}{formats}"
