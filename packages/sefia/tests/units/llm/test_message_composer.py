from unittest.mock import AsyncMock

from typing_extensions import override

from sefia.inference import FunctionInfo
from sefia.llm import (
    LLMClient,
    LLMInferenceStrategy,
    MarkdownPromptRenderer,
    Message,
    MessageComposer,
    MessageLayout,
)
from sefia.llm.transports import StructuredDecisionTransport
from sefia.pydantic import PydanticModelBackend
from sefia.pydantic._json_utils import pydantic_json_default
from sefia.testing import make_function_info


def _strategy(*composers: MessageComposer) -> LLMInferenceStrategy:
    return LLMInferenceStrategy(
        AsyncMock(spec=LLMClient),
        PydanticModelBackend(),
        MarkdownPromptRenderer(json_default=pydantic_json_default),
        StructuredDecisionTransport(),
        message_composers=composers,
    )


class _Append(MessageComposer):
    def __init__(self, name: str, received: list[MessageLayout]) -> None:
        self.name = name
        self.received = received

    @override
    async def compose(
        self, function: FunctionInfo, layout: MessageLayout
    ) -> MessageLayout:
        self.received.append(layout)
        return MessageLayout(
            before=layout.before,
            arguments=layout.arguments,
            after=(*layout.after, Message(role="user", content=self.name)),
        )


async def test_pipeline_passes_each_result_to_the_next_composer() -> None:
    received: list[MessageLayout] = []
    function = make_function_info(bound_arguments={"topic": "sefia"})
    layout = await _strategy(
        _Append("A", received), _Append("B", received), _Append("C", received)
    )._compose_message_layout(function)

    assert received[0] == MessageLayout.default(function)
    assert received[1].after[-1] == Message(role="user", content="A")
    assert received[2].after[-1] == Message(role="user", content="B")
    assert [message.content for message in layout.after] == ["A", "B", "C"]


class _Skip(MessageComposer):
    @override
    async def compose(
        self, function: FunctionInfo, layout: MessageLayout
    ) -> MessageLayout:
        return layout


async def test_non_applicable_composer_keeps_layout() -> None:
    function = make_function_info()
    assert await _strategy(_Skip())._compose_message_layout(function) == (
        MessageLayout.default(function)
    )
