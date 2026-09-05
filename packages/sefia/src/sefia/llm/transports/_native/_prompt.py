from typing import cast

from ....inference import HistoryItem, ToolCallsDecision
from ..._messages import Message, ToolCall
from ..._prompt_renderer import PromptRenderer
from ...structured_data import StructuredData, StructuredDataTree
from ...step_decision import DecisionSpec, StepDecisionMode, StepTool
from .._base import DecisionRequest


def render_native_prompt(
    request: DecisionRequest,
    renderer: PromptRenderer,
    result_tool: StepTool | None,
) -> str:
    return renderer.render(
        request.to_prompt(
            _response_instructions(request.decision_spec, result_tool),
            tools=(),
            history=(),
        )
    )


def _response_instructions(
    spec: DecisionSpec,
    result_tool: StepTool | None,
) -> str:
    if spec.mode is StepDecisionMode.TOOLS_REQUIRED:
        action = "Call one or more available tools."
    elif spec.mode is StepDecisionMode.RESULT_ONLY:
        assert result_tool is not None
        action = f"Call `{result_tool.name}` with the final result."
    else:
        assert result_tool is not None
        action = (
            "Call available tools when needed. When the task is complete, "
            f"call `{result_tool.name}` with the final result."
        )
    return f"{action}\nDo not describe a tool call or answer with text."


def native_history_messages(
    history: tuple[HistoryItem, ...],
    renderer: PromptRenderer,
) -> list[Message]:
    messages: list[Message] = []
    for item in history:
        if isinstance(item, ToolCallsDecision):
            messages.append(
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id=call.id,
                            name=call.name,
                            arguments=StructuredData.from_tree(
                                cast(StructuredDataTree, call.arguments)
                            ),
                        )
                        for call in item.calls
                    ],
                )
            )
        else:
            messages.append(
                Message(
                    role="tool",
                    content=renderer.render_tool_result(item),
                    tool_call_id=item.tool_call_id,
                )
            )
    return messages
