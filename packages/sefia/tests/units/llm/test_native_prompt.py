from unittest.mock import Mock

from sefia.inference import ToolCallResult, ToolCallsDecision
from sefia.llm import PromptRenderer, ToolCall
from sefia.llm.step_decision import DecisionSpec, StepDecisionMode, StepTool
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports._native._prompt import (
    native_history_messages,
    native_response_instructions,
)
from sefia.testing import make_tool_call_request


def test_native_history_messages() -> None:
    call = make_tool_call_request(
        id="call-1", name="lookup", arguments={"key": "first"}
    )
    result = ToolCallResult(tool_call_id=call.id, result={"value": "found"})
    renderer = Mock(spec=PromptRenderer)
    renderer.render_tool_result.return_value = '{"value": "found"}'

    messages = native_history_messages((ToolCallsDecision([call]), result), renderer)

    assert [message.role for message in messages] == ["assistant", "tool"]
    assert messages[0].tool_calls == [
        ToolCall(
            id=call.id,
            name=call.name,
            arguments=StructuredData.from_json({"key": "first"}),
        )
    ]
    renderer.render_tool_result.assert_called_once_with(result)
    assert messages[1].tool_call_id == call.id
    assert messages[1].content == '{"value": "found"}'


def test_native_response_instructions_use_selected_result_tool_name() -> None:
    spec = Mock(spec=DecisionSpec, mode=StepDecisionMode.TOOLS_OR_RESULT)
    result_tool = Mock(spec=StepTool)
    result_tool.name = "return_result_2"
    instructions = native_response_instructions(spec, result_tool)

    assert "return_result_2" in instructions
