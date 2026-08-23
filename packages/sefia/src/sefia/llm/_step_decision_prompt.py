from .step_decision import StepDecisionMode, StepDecisionSpec


def build_step_decision_prompt(spec: StepDecisionSpec) -> str:
    if spec.mode is StepDecisionMode.TOOLS_REQUIRED:
        instruction = (
            "Call one or more tools by setting `decision` to `tool_calls` and "
            "populating `tool_calls`; do not return a final result. Treat tool "
            "results as authoritative."
        )
    elif spec.mode is StepDecisionMode.TOOLS_OR_RESULT:
        instruction = (
            "Set `decision` to `tool_calls` and populate `tool_calls` when tools "
            "are needed; otherwise, set `decision` to `result` and populate "
            "`result` only when the task is complete. Treat tool results as "
            "authoritative."
        )
    else:
        instruction = (
            "Set `decision` to `result` and populate `result` with the non-null "
            "task result."
        )
    tools = ""
    if spec.tools:
        descriptions = [
            f"- `{tool.name}`: {tool.definition().description}" for tool in spec.tools
        ]
        tools = "\n\nAvailable tools:\n" + "\n".join(descriptions)
    return f"\n\n{instruction}{tools}"
