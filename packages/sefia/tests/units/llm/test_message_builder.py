import json
from unittest.mock import Mock

from sefia.exceptions import InvalidInferenceResponseError
from sefia.inference import (
    FunctionInfo,
    ToolCallsDecision,
    ToolCallRequest,
    ToolCallResult,
)
from sefia.llm import PromptRenderer
from sefia.llm._message_builder import (
    build_messages,
    build_response_feedback_messages,
)
from sefia.llm.step_decision import StepDecisionSpec
from sefia.pydantic._json_utils import pydantic_json_default


def _function_info() -> FunctionInfo:
    return FunctionInfo(
        qualname="test",
        instructions="instructions",
        bound_arguments={"arg": "value"},
        type_hints={},
        return_type=str,
        args=(),
        kwargs={},
    )


def _decision_spec() -> StepDecisionSpec:
    return StepDecisionSpec.for_inference(
        name="StepDecision",
        output_type=str,
        tools=[],
    )


def _prompt_renderer() -> Mock:
    renderer = Mock(spec=PromptRenderer)
    renderer.render_instructions.return_value = "rendered instructions"
    renderer.render_invocation.return_value = "rendered invocation"
    renderer.render_response_feedback.return_value = "rendered feedback"
    return renderer


def test_build_messages_assigns_prompt_content_to_message_roles():
    renderer = _prompt_renderer()

    messages = build_messages(
        _function_info(),
        [],
        _decision_spec(),
        renderer,
        pydantic_json_default,
    )

    assert [(message.role, message.content) for message in messages] == [
        ("system", "rendered instructions"),
        ("user", "rendered invocation"),
    ]
    renderer.render_instructions.assert_called_once_with(
        _function_info(),
        _decision_spec(),
    )
    renderer.render_invocation.assert_called_once_with(_function_info())


def test_build_messages_replays_history_without_involving_the_prompt_renderer():
    renderer = _prompt_renderer()
    history = [
        ToolCallsDecision(
            calls=[
                ToolCallRequest(
                    id="1",
                    name="search",
                    arguments={"q": "日本語の検索クエリ"},
                )
            ]
        ),
        ToolCallResult(tool_call_id="1", result="見つかりました"),
    ]

    messages = build_messages(
        _function_info(),
        history,
        _decision_spec(),
        renderer,
        pydantic_json_default,
    )

    assert json.loads(str(messages[2].content)) == {
        "decision": "tool_calls",
        "tool_calls": [
            {
                "id": "1",
                "name": "search",
                "arguments": {"q": "日本語の検索クエリ"},
            }
        ],
    }
    assert "\\u65e5" not in str(messages[2].content)
    assert json.loads(str(messages[3].content)) == {
        "tool_call_result": {
            "tool_call_id": "1",
            "result": "見つかりました",
        }
    }
    assert "\\u898b" not in str(messages[3].content)


def test_build_response_feedback_messages_echoes_invalid_output():
    renderer = _prompt_renderer()
    error = InvalidInferenceResponseError(
        "invalid schema",
        raw_content="invalid",
    )

    messages = build_response_feedback_messages(error, renderer)

    assert [(message.role, message.content) for message in messages] == [
        ("assistant", "invalid"),
        ("user", "rendered feedback"),
    ]
    renderer.render_response_feedback.assert_called_once_with(error)


def test_build_response_feedback_messages_does_not_echo_an_empty_response():
    renderer = _prompt_renderer()

    messages = build_response_feedback_messages(
        InvalidInferenceResponseError("empty response"),
        renderer,
    )

    assert [(message.role, message.content) for message in messages] == [
        ("user", "rendered feedback")
    ]
