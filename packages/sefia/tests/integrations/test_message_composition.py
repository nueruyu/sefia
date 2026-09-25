import json
from unittest.mock import AsyncMock

import pytest
from typing_extensions import override

from sefia import ToolRegistry
from sefia.event_system import EventPublisher
from sefia.inference import (
    FunctionInfo,
    ResultDecision,
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
    DecisionRequest,
    NativeDecisionTransport,
    PromptedDecisionTransport,
    DecisionToolCalls,
    DecisionToolResult,
    RejectedDecision,
    StructuredDecisionTransport,
)
from sefia.pydantic import (
    PydanticResultFormatFactory,
    PydanticStructuredDataConverter,
)
from sefia.testing import (
    RecordingDecisionObserver,
    make_decision_request,
    make_function_info,
)


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
        PydanticResultFormatFactory(),
        PydanticStructuredDataConverter(),
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
        output_type=str, tools=[], result_format_factory=PydanticResultFormatFactory()
    )


def _spec_with_tool() -> DecisionSpec:
    registry = ToolRegistry()

    def lookup(query: str) -> str:
        """Look up a value."""
        return query

    registry.add(lookup, name="lookup")
    return DecisionSpec.for_inference(
        output_type=str,
        tools=registry.get_all(),
        result_format_factory=PydanticResultFormatFactory(),
    )


def _renderer() -> MarkdownPromptRenderer:
    return MarkdownPromptRenderer()


def _request_with_application_messages() -> tuple[DecisionRequest, tuple[Message, ...]]:
    function = make_function_info(bound_arguments={"knowledge": {"foo": "bar"}})
    application_messages = (
        Message(role="developer", content="Reply briefly."),
        Message(role="user", content="old question"),
        Message(role="assistant", content="old answer"),
        Message(role="user", content="current question"),
    )
    history = (
        DecisionToolCalls(
            (
                ToolCall(
                    id="call-1",
                    name="lookup",
                    arguments=StructuredData.from_object({}),
                ),
            )
        ),
        DecisionToolResult(
            tool_call_id="call-1",
            result=StructuredData.from_json({"value": "found"}),
        ),
    )
    rejected = RejectedDecision(
        completion=LLMCompletion(content="bad"),
        reason="invalid",
    )
    return (
        make_decision_request(
            _spec(),
            function=function,
            arguments=StructuredData.from_json({"knowledge": {"foo": "bar"}}),
            messages_before=(application_messages[0],),
            messages_after=application_messages[1:],
            history=history,
            rejected=rejected,
        ),
        application_messages,
    )


def _assert_application_prefix(
    messages: list[Message], application_messages: tuple[Message, ...]
) -> None:
    assert messages[0] is application_messages[0]
    prompt_content = messages[1].content
    assert isinstance(prompt_content, str)
    assert "## Task arguments" in prompt_content
    assert '"foo": "bar"' in prompt_content
    assert "## Response" not in prompt_content
    assert all(
        messages[index + 2] is message
        for index, message in enumerate(application_messages[1:])
    )


@pytest.mark.parametrize(
    ("transport", "completion", "passes_decision_spec"),
    [
        (
            StructuredDecisionTransport(),
            LLMCompletion(
                structured_output=StructuredData.from_json(
                    {"decision": "result", "result": "done"}
                )
            ),
            True,
        ),
        (
            PromptedDecisionTransport(),
            LLMCompletion(content='{"decision":"result","result":"done"}'),
            False,
        ),
    ],
)
async def test_text_transport_appends_text_history_after_application_messages(
    transport: StructuredDecisionTransport | PromptedDecisionTransport,
    completion: LLMCompletion,
    passes_decision_spec: bool,
) -> None:
    request, application_messages = _request_with_application_messages()
    client = AsyncMock()
    client.complete.return_value = completion
    observer = RecordingDecisionObserver()

    decoded = await transport.request_decision(
        client,
        _renderer(),
        request,
        observer,
        False,
    )

    assert decoded.decision_data.tree == {"decision": "result", "result": "done"}
    sent = client.complete.await_args.kwargs
    messages = sent["messages"]
    assert observer.messages == tuple(messages)
    _assert_application_prefix(messages, application_messages)
    history_content = messages[5].content
    assert isinstance(history_content, str)
    assert "Previous tool interactions" in history_content
    assert '"value": "found"' in history_content
    repair_content = messages[-2].content
    response_content = messages[-1].content
    assert isinstance(repair_content, str)
    assert isinstance(response_content, str)
    assert "Correct the previous response" in repair_content
    assert "## Response" in response_content
    assert sent["decision_spec"] is (
        request.decision_spec if passes_decision_spec else None
    )


async def test_native_transport_appends_native_history_after_application_messages() -> (
    None
):
    request, application_messages = _request_with_application_messages()
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        tool_calls=[
            ToolCall(
                id="result-1",
                name="return_result",
                arguments=StructuredData.from_json({"result": "done"}),
            )
        ]
    )

    decoded = await NativeDecisionTransport().request_decision(
        client,
        _renderer(),
        request,
        RecordingDecisionObserver(),
        False,
    )

    assert decoded.decision_data.tree == {"decision": "result", "result": "done"}
    sent = client.complete.await_args.kwargs
    messages = sent["messages"]
    _assert_application_prefix(messages, application_messages)
    assert [message.role for message in messages[5:7]] == ["assistant", "tool"]
    calls = messages[5].tool_calls
    assert calls is not None
    assert calls[0].id == messages[6].tool_call_id == "call-1"
    tool_content = messages[6].content
    assert isinstance(tool_content, str)
    assert json.loads(tool_content) == {"value": "found"}
    assert sent["tools"][0].name == "return_result"
    assert sent["decision_spec"] is None
    repair_content = messages[-2].content
    response_content = messages[-1].content
    assert isinstance(repair_content, str)
    assert isinstance(response_content, str)
    assert "Correct the previous response" in repair_content
    assert "## Response" in response_content


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
    ],
)
async def test_text_transport_default_layout_sends_one_message_on_first_step(
    transport: StructuredDecisionTransport | PromptedDecisionTransport,
    completion: LLMCompletion,
) -> None:
    function = make_function_info(bound_arguments={"topic": "sefia"})
    request = make_decision_request(
        _spec(),
        function=function,
        arguments=StructuredData.from_json({"topic": "sefia"}),
    )
    client = AsyncMock()
    client.complete.return_value = completion

    await transport.request_decision(
        client,
        _renderer(),
        request,
        RecordingDecisionObserver(),
        False,
    )

    messages = client.complete.await_args.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0].role == "user"
    content = messages[0].content
    assert isinstance(content, str)
    assert '"topic": "sefia"' in content
    assert "## Response" in content


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
    ],
)
async def test_text_transport_puts_decision_tools_in_the_inference_prompt(
    transport: StructuredDecisionTransport | PromptedDecisionTransport,
    completion: LLMCompletion,
) -> None:
    request = make_decision_request(_spec_with_tool())
    client = AsyncMock()
    client.complete.return_value = completion
    renderer = _CapturingRenderer()

    await transport.request_decision(
        client,
        renderer,
        request,
        RecordingDecisionObserver(),
        False,
    )

    assert renderer.prompts[0].tools == request.decision_spec.tools


async def test_native_transport_default_layout_sends_one_message_on_first_step() -> (
    None
):
    request = make_decision_request(
        _spec(),
        function=make_function_info(bound_arguments={"topic": "sefia"}),
        arguments=StructuredData.from_json({"topic": "sefia"}),
    )
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        tool_calls=[
            ToolCall(
                "result-1",
                "return_result",
                StructuredData.from_json({"result": "done"}),
            )
        ]
    )

    await NativeDecisionTransport().request_decision(
        client,
        _renderer(),
        request,
        RecordingDecisionObserver(),
        False,
    )

    messages = client.complete.await_args.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0].role == "user"
    content = messages[0].content
    assert isinstance(content, str)
    assert '"topic": "sefia"' in content
    assert "## Response" in content


async def test_text_and_native_transports_share_repair_framing() -> None:
    request = make_decision_request(
        _spec(),
        rejected=RejectedDecision(
            completion=LLMCompletion(content="invalid response"),
            reason="invalid decision",
        ),
    )
    text_client = AsyncMock()
    text_client.complete.return_value = LLMCompletion(
        structured_output=StructuredData.from_json(
            {"decision": "result", "result": "done"}
        )
    )
    native_client = AsyncMock()
    native_client.complete.return_value = LLMCompletion(
        tool_calls=[
            ToolCall(
                id="result-1",
                name="return_result",
                arguments=StructuredData.from_json({"result": "done"}),
            )
        ]
    )

    await StructuredDecisionTransport().request_decision(
        text_client,
        _renderer(),
        request,
        RecordingDecisionObserver(),
        False,
    )
    await NativeDecisionTransport().request_decision(
        native_client,
        _renderer(),
        request,
        RecordingDecisionObserver(),
        False,
    )

    text_messages = text_client.complete.await_args.kwargs["messages"]
    native_messages = native_client.complete.await_args.kwargs["messages"]
    assert text_messages[-2] == native_messages[-2]
    repair = text_messages[-2].content
    assert isinstance(repair, str)
    assert "Correct the previous response" in repair


async def test_custom_prompt_renderer_receives_materialized_request() -> None:
    request = make_decision_request(
        _spec(),
        arguments=StructuredData.from_json({"payload": {"value": "argument"}}),
        history=(
            DecisionToolResult(
                tool_call_id="call-1",
                result=StructuredData.from_json({"value": "history"}),
            ),
        ),
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
    )

    assert renderer.prompts[0].arguments.tree == {"payload": {"value": "argument"}}
    messages = client.complete.await_args.kwargs["messages"]
    assert messages[0].content == "custom inference prompt"
    assert '"value": "history"' in messages[1].content


async def test_repair_reuses_immutable_application_message() -> None:
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
        observed.append(messages[0])
        content = messages[0].content
        assert isinstance(content, list)
        content[0]["text"] = "changed locally"
        messages[0] = Message(role="user", content="replacement")
        return completions.pop(0)

    client.complete.side_effect = complete
    strategy = LLMInferenceStrategy(
        client,
        PydanticResultFormatFactory(),
        PydanticStructuredDataConverter(),
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
    assert all(message is source for message in observed)
    assert source.content == [{"text": "original"}]
    assert len(layouts) == 1
    assert layouts[0].before[0] is source
