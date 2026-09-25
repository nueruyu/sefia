from unittest.mock import Mock

from sefia.llm import ToolCall
from sefia.llm.json import JsonSnapshot
from sefia.llm.step_decision import DecisionSpec, StepDecisionMode, StepTool
from sefia.llm.transports import DecisionToolCalls, DecisionToolResult
from sefia.llm.transports._native._prompt import (
    native_history_messages,
    native_response_instructions,
)


def test_native_history_messages() -> None:
    call = ToolCall(
        id="call-1",
        name="lookup",
        arguments=JsonSnapshot.capture({"key": "first"}),
    )
    result = DecisionToolResult(
        tool_call_id=call.id,
        result=JsonSnapshot.capture({"value": "found"}),
    )
    messages = native_history_messages((DecisionToolCalls((call,)), result))

    assert [message.role for message in messages] == ["assistant", "tool"]
    assert messages[0].tool_calls == (
        ToolCall(
            id=call.id,
            name=call.name,
            arguments=JsonSnapshot.capture({"key": "first"}),
        ),
    )
    assert messages[0].tool_calls is not None
    assert messages[0].tool_calls[0] is call
    assert messages[1].tool_call_id == call.id
    assert messages[1].content == '{"value":"found"}'


def test_native_response_instructions_use_selected_result_tool_name() -> None:
    spec = Mock(spec=DecisionSpec, mode=StepDecisionMode.TOOLS_OR_RESULT)
    result_tool = Mock(spec=StepTool)
    result_tool.name = "return_result_2"
    instructions = native_response_instructions(spec, result_tool)

    assert "return_result_2" in instructions
