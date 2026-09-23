import json
from copy import deepcopy
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from typing_extensions import override

from sefia import ToolRegistry
from sefia.event_system import EventPublisher
from sefia.inference import (
    FunctionInfo,
    ResultDecision,
    ToolCallResult,
    ToolCallsDecision,
)
from sefia.llm import (
    LLMClient,
    LLMCompletion,
    LLMInferenceStrategy,
    InferencePrompt,
    MarkdownPromptRenderer,
    Message,
    MessageComposer,
    MessageLayout,
    PromptRenderer,
    ToolCall,
)
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports import (
    NativeDecisionTransport,
    PromptedDecisionTransport,
    RejectedDecision,
    StructuredDecisionTransport,
)
from sefia.pydantic import PydanticModelBackend
from sefia.testing import (
    RecordingDecisionObserver,
    make_decision_request,
    make_function_info,
    make_tool_call_request,
)


@dataclass(frozen=True)
class _StructuredResult:
    value: str


class _CapturingRenderer(PromptRenderer):
    def __init__(self) -> None:
        self.prompts: list[InferencePrompt] = []

    @override
    def render(self, prompt: InferencePrompt) -> str:
        self.prompts.append(prompt)
        return "custom inference prompt"


class _Layer(MessageComposer):
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    @override
    async def compose(
        self, function: FunctionInfo, layout: MessageLayout
    ) -> MessageLayout:
        assert function.bound_arguments["question"] == "current"
        assert function.prompt_arguments["knowledge"] == {"a": 1}
        assert function.type_hints["question"] is str
        assert function.instructions == "Answer the question."
        self.log.append(self.name)
        return MessageLayout(
            before=(*layout.before, Message(role="developer", content=self.name)),
            arguments={"knowledge": layout.arguments["knowledge"]},
            after=layout.after,
        )


async def test_message_composers_transform_in_declared_order() -> None:
    function = make_function_info(
        instructions="Answer the question.",
        bound_arguments={"question": "current", "knowledge": {"a": 1}},
        type_hints={"question": str},
    )
    log: list[str] = []
    strategy = LLMInferenceStrategy(
        AsyncMock(spec=LLMClient),
        PydanticModelBackend(),
        _renderer(),
        StructuredDecisionTransport(),
        message_composers=(_Layer("A", log), _Layer("B", log), _Layer("C", log)),
    )
    layout = await strategy._compose_message_layout(function)

    assert log == ["A", "B", "C"]
    assert [message.content for message in layout.before] == [
        "A",
        "B",
        "C",
    ]
    assert layout.arguments == {"knowledge": {"a": 1}}
    assert layout.after == ()


def _spec() -> DecisionSpec:
    return DecisionSpec.for_inference(
        output_type=str, tools=[], result_format_factory=PydanticModelBackend()
    )


def _renderer() -> MarkdownPromptRenderer:
    return MarkdownPromptRenderer()


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
    layout = MessageLayout(
        before=(application_messages[0],),
        arguments=function.prompt_arguments,
        after=application_messages[1:],
    )
    history = (
        ToolCallsDecision([make_tool_call_request(id="call-1", name="lookup")]),
        ToolCallResult(tool_call_id="call-1", result=_StructuredResult("found")),
    )
    rejected = RejectedDecision(content="bad", reason="invalid")
    request = make_decision_request(
        _spec(),
        function=function,
        message_layout=layout,
        history=history,
        rejected=rejected,
    )
    client = AsyncMock()
    client.complete.return_value = completion
    observer = RecordingDecisionObserver()

    decoded = await transport.request_decision(
        client,
        _renderer(),
        request,
        observer,
        False,
        dump=PydanticModelBackend().dump,
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
        assert json.loads(messages[6].content) == {"value": "found"}
        assert sent["tools"][0].name == "return_result"
    else:
        assert messages[5].role == "user"
        assert "Previous tool interactions" in messages[5].content
        assert '"value": "found"' in messages[5].content
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
async def test_default_layout_sends_one_inference_message_on_first_step(
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
        client,
        _renderer(),
        request,
        RecordingDecisionObserver(),
        False,
        dump=PydanticModelBackend().dump,
    )

    messages = client.complete.await_args.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert '"topic": "sefia"' in messages[0].content
    assert "## Response" in messages[0].content


async def test_transport_dumps_raw_values_independently_of_prompt_renderer() -> None:
    argument = _StructuredResult("argument")
    result = _StructuredResult("history")
    function = make_function_info(bound_arguments={"payload": argument})
    request = make_decision_request(
        _spec(),
        function=function,
        history=(ToolCallResult(tool_call_id="call-1", result=result),),
    )
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        structured_output=StructuredData.from_json(
            {"decision": "result", "result": "done"}
        )
    )
    renderer = _CapturingRenderer()

    await StructuredDecisionTransport().request_decision(
        client,
        renderer,
        request,
        RecordingDecisionObserver(),
        False,
        dump=PydanticModelBackend().dump,
    )

    assert renderer.prompts[0].arguments.tree == {"payload": {"value": "argument"}}
    messages = client.complete.await_args.kwargs["messages"]
    assert messages[0].content == "custom inference prompt"
    assert '"value": "history"' in messages[1].content
    assert request.message_layout.arguments["payload"] is argument
    history_item = request.history[0]
    assert isinstance(history_item, ToolCallResult)
    assert history_item.result is result


async def test_client_mutation_does_not_change_layout_used_for_repair() -> None:
    source = Message(role="developer", content=[{"text": "original"}])
    layouts: list[MessageLayout] = []

    class _SourceComposer(MessageComposer):
        @override
        async def compose(
            self, function: FunctionInfo, layout: MessageLayout
        ) -> MessageLayout:
            composed = MessageLayout(before=(source,), arguments={}, after=())
            layouts.append(composed)
            return composed

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
        client,
        PydanticModelBackend(),
        _renderer(),
        StructuredDecisionTransport(),
        message_composers=(_SourceComposer(),),
    )

    decision = await strategy.decide_next_step(
        make_function_info(), [], ToolRegistry(), EventPublisher([])
    )

    assert isinstance(decision, ResultDecision)
    assert decision.result == "done"
    assert observed == [source, source]
    assert all(message is not source for message in observed)
    assert source.content == [{"text": "original"}]
    assert len(layouts) == 1
    assert layouts[0].before[0] is source
