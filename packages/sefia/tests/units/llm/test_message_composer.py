from typing import cast
from unittest.mock import AsyncMock

import pytest
from typing_extensions import override

from sefia.inference import FunctionInfo
from sefia.llm import (
    LLMClient,
    LLMInferenceStrategy,
    MarkdownPromptRenderer,
    Message,
    MessageComposer,
    MessagePlan,
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
    def __init__(self, name: str, received: list[MessagePlan]) -> None:
        self.name = name
        self.received = received

    @override
    async def compose(self, function: FunctionInfo, plan: MessagePlan) -> MessagePlan:
        self.received.append(plan)
        return MessagePlan(parts=(*plan.parts, Message(role="user", content=self.name)))


async def test_pipeline_passes_each_result_to_the_next_composer() -> None:
    received: list[MessagePlan] = []
    function = make_function_info(bound_arguments={"topic": "sefia"})
    plan = await _strategy(
        _Append("A", received), _Append("B", received), _Append("C", received)
    )._compose_message_plan(function)

    assert received[0] == MessagePlan.default(function)
    assert received[1].parts[-1] == Message(role="user", content="A")
    assert received[2].parts[-1] == Message(role="user", content="B")
    assert [part.content for part in plan.parts if isinstance(part, Message)] == [
        "A",
        "B",
        "C",
    ]


class _Skip(MessageComposer):
    @override
    async def compose(self, function: FunctionInfo, plan: MessagePlan) -> MessagePlan:
        return plan


async def test_non_applicable_composer_keeps_plan() -> None:
    function = make_function_info()
    assert await _strategy(_Skip())._compose_message_plan(function) == (
        MessagePlan.default(function)
    )


class _Invalid(MessageComposer):
    @override
    async def compose(self, function: FunctionInfo, plan: MessagePlan) -> MessagePlan:
        return cast(MessagePlan, "invalid")


async def test_invalid_composer_result_names_offending_composer() -> None:
    with pytest.raises(
        TypeError, match=r"_Invalid.compose\(\) must return MessagePlan"
    ):
        await _strategy(_Invalid())._compose_message_plan(make_function_info())


def test_strategy_rejects_non_composer_configuration() -> None:
    with pytest.raises(
        TypeError, match=r"message_composers\[0\] must be MessageComposer"
    ):
        _strategy(cast(MessageComposer, object()))
