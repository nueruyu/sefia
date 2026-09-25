from collections.abc import Callable
from unittest.mock import AsyncMock, Mock

import pytest
from sefia.llm import LLMClient, LLMCompletion, LLMInferenceStrategy, PromptRenderer
from sefia.llm.json import JsonSnapshot
from sefia.llm.transports import DecisionTransport, DecodedDecision
from sefia.pydantic import (
    PydanticJsonMaterializer,
    PydanticResultFormatFactory,
)


@pytest.fixture
def transport() -> AsyncMock:
    transport = AsyncMock(spec=DecisionTransport)
    transport.request_decision.return_value = DecodedDecision(
        JsonSnapshot.capture({"decision": "result", "result": "done"}),
        LLMCompletion(content="done"),
    )
    return transport


@pytest.fixture
def make_strategy(transport: AsyncMock) -> Callable[..., LLMInferenceStrategy]:
    def factory(
        *, stream: bool = False, max_repair_attempts: int = 2
    ) -> LLMInferenceStrategy:
        return LLMInferenceStrategy(
            llm_client=Mock(spec=LLMClient),
            result_format_factory=PydanticResultFormatFactory(),
            json_materializer=PydanticJsonMaterializer(),
            prompt_renderer=Mock(spec=PromptRenderer),
            decision_transport=transport,
            stream=stream,
            max_repair_attempts=max_repair_attempts,
        )

    return factory
