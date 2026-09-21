from unittest.mock import AsyncMock, Mock

import pytest
from typing_extensions import override

from sefia.llm import InferencePrompt, LLMCompletion, Message, PromptRenderer
from sefia.llm.exceptions import DecisionDecodingError
from sefia.llm.structured_data import StructuredData
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.transports import DecisionRequest, StructuredDecisionTransport
from sefia.pydantic import PydanticModelBackend
from sefia.testing import RecordingDecisionObserver, make_decision_request


def _request() -> DecisionRequest:
    decision = DecisionSpec.for_inference(
        output_type=str,
        tools=[],
        result_format_factory=PydanticModelBackend(),
    )
    return make_decision_request(decision)


def _renderer(prompt: str = "complete prompt") -> Mock:
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = prompt
    return renderer


async def test_renders_and_delivers_one_complete_prompt() -> None:
    client = AsyncMock()
    completion = LLMCompletion(
        structured_output=StructuredData.from_json(
            {"decision": "result", "result": "done"}
        )
    )
    client.complete.return_value = completion
    renderer = _renderer()
    request = _request()
    observer = RecordingDecisionObserver()

    decoded = await StructuredDecisionTransport().request_decision(
        client, renderer, request, observer, stream=False
    )

    sent = client.complete.await_args.kwargs
    assert len(sent["messages"]) == 1
    assert sent["messages"][0].role == "user"
    content = sent["messages"][0].content
    assert content.startswith("complete prompt\n\n## Response\n\n")
    assert sent["decision_spec"] is request.decision_spec
    assert observer.messages == tuple(sent["messages"])
    assert decoded.decision_data.tree == {"decision": "result", "result": "done"}
    assert decoded.completion is completion
    renderer.render.assert_called_once()
    assert isinstance(renderer.render.call_args.args[0], InferencePrompt)
    assert "provided structured output schema" in content
    assert '"decision"' not in content


async def test_observer_finishes_before_the_client_request() -> None:
    order: list[str] = []
    client = AsyncMock()

    async def complete(**_kwargs: object) -> LLMCompletion:
        order.append("request")
        return LLMCompletion(
            structured_output=StructuredData.from_json(
                {"decision": "result", "result": "done"}
            )
        )

    client.complete.side_effect = complete

    class Observer(RecordingDecisionObserver):
        @override
        async def before_request(self, messages: tuple[Message, ...]) -> None:
            await super().before_request(messages)
            order.append("observed")

    await StructuredDecisionTransport().request_decision(
        client, _renderer(), _request(), Observer(), stream=False
    )

    assert order == ["observed", "request"]


async def test_reports_undecodable_response() -> None:
    client = AsyncMock()
    completion = LLMCompletion(content="not json")
    client.complete.return_value = completion

    with pytest.raises(DecisionDecodingError) as exc_info:
        await StructuredDecisionTransport().request_decision(
            client,
            _renderer(),
            _request(),
            RecordingDecisionObserver(),
            stream=False,
        )

    assert exc_info.value.completion is completion


async def test_rejects_raw_json_content() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        content='{"decision":"result","result":"done"}'
    )

    with pytest.raises(DecisionDecodingError, match="structured output"):
        await StructuredDecisionTransport().request_decision(
            client,
            _renderer(),
            _request(),
            RecordingDecisionObserver(),
            stream=False,
        )
