from .step_decision import StepDecisionMode, StepDecisionSpec


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
    tools = ""
    if spec.tools:
        instruction += (
            " Treat all tool interactions in the conversation history as part of "
            "this decision protocol. Request new tool calls only through this "
            "response schema."
        )
        descriptions = [
            f"- `{tool.name}`: {tool.definition().description}" for tool in spec.tools
        ]
        tools = "\n\nAvailable tools:\n" + "\n".join(descriptions)
    return f"\n\n{instruction}{tools}"
