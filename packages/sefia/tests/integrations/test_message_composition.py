from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import cast
from unittest.mock import AsyncMock

import pytest
from typing_extensions import override

from sefia import MessageContext, MessageMiddleware, MessagePlan, TaskPrompt
from sefia import ToolRegistry
from sefia._executor import _compose
from sefia.event_system import EventPublisher
from sefia.inference import ResultDecision, ToolCallResult, ToolCallsDecision
from sefia.llm import (
    LLMClient,
    LLMCompletion,
    LLMInferenceStrategy,
    MarkdownPromptRenderer,
    Message,
    ToolCall,
)
from sefia.llm._prompt_renderer import RejectedDecision
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports import (
    NativeDecisionTransport,
    PromptedDecisionTransport,
    StructuredDecisionTransport,
)
from sefia.pydantic import PydanticModelBackend
from sefia.pydantic._json_utils import pydantic_json_default
from sefia.testing import (
    RecordingDecisionObserver,
    make_decision_request,
    make_function_info,
    make_tool_call_request,
)


class _Layer(MessageMiddleware):
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    @override
    async def wrap(
        self, ctx: MessageContext, nxt: Callable[[], Awaitable[MessagePlan]]
    ) -> MessagePlan:
        assert ctx.step == 3
        assert ctx.function.bound_arguments["question"] == "current"
        self.log.append(f"{self.name}:enter")
        plan = await nxt()
        self.log.append(f"{self.name}:exit")
        task = next(part for part in plan.parts if isinstance(part, TaskPrompt))
        return MessagePlan(
            parts=(
                Message(role="developer", content=self.name),
                *(part for part in plan.parts if part is not task),
                TaskPrompt(arguments={"knowledge": task.arguments["knowledge"]}),
            )
        )


async def test_message_middleware_uses_onion_order_and_function_context() -> None:
    function = make_function_info(
        bound_arguments={"question": "current", "knowledge": {"a": 1}}
    )
    log: list[str] = []

    async def core() -> MessagePlan:
        log.append("core")
        return MessagePlan.default(function)

    plan = await _compose(
        [_Layer("outer", log), _Layer("inner", log)],
        MessageContext(step=3, function=function),
        core,
    )()

    assert log == ["outer:enter", "inner:enter", "core", "inner:exit", "outer:exit"]
    assert [cast(Message, part).content for part in plan.parts[:-1]] == [
        "outer",
        "inner",
    ]
    assert plan.parts[-1] == TaskPrompt(arguments={"knowledge": {"a": 1}})


def _spec() -> DecisionSpec:
    return DecisionSpec.for_inference(
        output_type=str, tools=[], result_format_factory=PydanticModelBackend()
    )


def _renderer() -> MarkdownPromptRenderer:
    return MarkdownPromptRenderer(json_default=pydantic_json_default)


@pytest.mark.parametrize(
    ("transport", "completion"),
    [
        (
            StructuredDecisionTransport(),
            LLMCompletion(
                structured_output=StructuredData.from_json(
                    {"decision": "result", "result": "done"}
                )
            ),
        ),
        (
            PromptedDecisionTransport(),
            LLMCompletion(content='{"decision":"result","result":"done"}'),
        ),
        (
            NativeDecisionTransport(),
            LLMCompletion(
                tool_calls=[
                    ToolCall(
                        id="result-1",
                        name="return_result",
                        arguments=StructuredData.from_json({"result": "done"}),
                    )
                ]
            ),
        ),
    ],
)
async def test_transport_preserves_application_plan_then_appends_history_and_repair(
    transport: StructuredDecisionTransport
    | PromptedDecisionTransport
    | NativeDecisionTransport,
    completion: LLMCompletion,
) -> None:
    function = make_function_info(bound_arguments={"knowledge": {"foo": "bar"}})
    application_messages = (
        Message(role="developer", content="Reply briefly."),
        Message(role="user", content="old question"),
        Message(role="assistant", content="old answer"),
        Message(role="user", content="current question"),
    )
    plan = MessagePlan(
        parts=(
            application_messages[0],
            TaskPrompt(arguments=function.prompt_arguments),
            *application_messages[1:],
        )
    )
    history = (
        ToolCallsDecision([make_tool_call_request(id="call-1", name="lookup")]),
        ToolCallResult(tool_call_id="call-1", result="found"),
    )
    rejected = RejectedDecision(content="bad", reason="invalid")
    request = make_decision_request(
        _spec(),
        function=function,
        message_plan=plan,
        history=history,
        rejected=rejected,
    )
    client = AsyncMock()
    client.complete.return_value = completion
    observer = RecordingDecisionObserver()

    decoded = await transport.request_decision(
        client, _renderer(), request, observer, False
    )

    assert decoded.decision_data.tree == {"decision": "result", "result": "done"}
    sent = client.complete.await_args.kwargs
    messages = sent["messages"]
    assert observer.messages == tuple(messages)
    assert messages[0] == application_messages[0]
    assert messages[0] is not application_messages[0]
    assert messages[1].role == "user"
    assert "## Task arguments" in messages[1].content
    assert '"foo": "bar"' in messages[1].content
    assert "## Response" not in messages[1].content
    assert all(
        messages[i + 2] == item and messages[i + 2] is not item
        for i, item in enumerate(application_messages[1:])
    )
    assert "Correct the previous response" in messages[-2].content
    assert "## Response" in messages[-1].content
    if isinstance(transport, NativeDecisionTransport):
        assert [message.role for message in messages[5:7]] == ["assistant", "tool"]
        assert messages[5].tool_calls[0].id == messages[6].tool_call_id == "call-1"
        assert sent["tools"][0].name == "return_result"
    else:
        assert messages[5].role == "user"
        assert "Previous tool interactions" in messages[5].content
        assert (
            sent["decision_spec"] is request.decision_spec
            if isinstance(transport, StructuredDecisionTransport)
            else sent["decision_spec"] is None
        )


@pytest.mark.parametrize(
    "transport",
    [
        StructuredDecisionTransport(),
        PromptedDecisionTransport(),
        NativeDecisionTransport(),
    ],
)
async def test_default_plan_sends_one_task_message_on_first_step(
    transport: StructuredDecisionTransport
    | PromptedDecisionTransport
    | NativeDecisionTransport,
) -> None:
    function = make_function_info(bound_arguments={"topic": "sefia"})
    request = make_decision_request(_spec(), function=function)
    client = AsyncMock()
    client.complete.return_value = (
        LLMCompletion(
            structured_output=StructuredData.from_json(
                {"decision": "result", "result": "done"}
            )
        )
        if isinstance(transport, StructuredDecisionTransport)
        else LLMCompletion(content='{"decision":"result","result":"done"}')
        if isinstance(transport, PromptedDecisionTransport)
        else LLMCompletion(
            tool_calls=[
                ToolCall(
                    "result-1",
                    "return_result",
                    StructuredData.from_json({"result": "done"}),
                )
            ]
        )
    )

    await transport.request_decision(
        client, _renderer(), request, RecordingDecisionObserver(), False
    )

    messages = client.complete.await_args.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert '"topic": "sefia"' in messages[0].content
    assert "## Response" in messages[0].content


async def test_transport_rejects_unknown_plan_part() -> None:
    plan = MessagePlan(
        parts=cast(tuple[Message | TaskPrompt, ...], (TaskPrompt({}), object()))
    )
    request = make_decision_request(_spec(), message_plan=plan)

    with pytest.raises(TypeError, match="Unknown MessagePlan part type: object"):
        await StructuredDecisionTransport().request_decision(
            AsyncMock(), _renderer(), request, RecordingDecisionObserver(), False
        )


async def test_client_mutation_does_not_change_plan_used_for_repair() -> None:
    source = Message(role="developer", content=[{"text": "original"}])
    plan = MessagePlan(parts=(source, TaskPrompt(arguments={})))
    client = AsyncMock(spec=LLMClient)
    observed: list[Message] = []
    completions = [
        LLMCompletion(content="invalid"),
        LLMCompletion(
            structured_output=StructuredData.from_json(
                {"decision": "result", "result": "done"}
            )
        ),
    ]

    async def complete(*, messages: list[Message], **_kwargs: object) -> LLMCompletion:
        observed.append(deepcopy(messages[0]))
        messages[0].role = "user"
        assert isinstance(messages[0].content, list)
        messages[0].content[0]["text"] = "changed by client"
        return completions.pop(0)

    client.complete.side_effect = complete
    strategy = LLMInferenceStrategy(
        client, PydanticModelBackend(), _renderer(), StructuredDecisionTransport()
    )

    decision = await strategy.decide_next_step(
        make_function_info(), plan, [], ToolRegistry(), EventPublisher([])
    )

    assert isinstance(decision, ResultDecision)
    assert decision.result == "done"
    assert observed == [source, source]
    assert all(message is not source for message in observed)
    assert source.content == [{"text": "original"}]
