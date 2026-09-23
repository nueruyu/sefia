from collections.abc import Callable
from unittest.mock import AsyncMock, Mock

import pytest
from sefia.llm import LLMClient, LLMCompletion, LLMInferenceStrategy, PromptRenderer
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports import DecisionTransport, DecodedDecision
from sefia.pydantic import (
    PydanticResultFormatFactory,
    PydanticStructuredDataConverter,
)


@pytest.fixture
def transport() -> AsyncMock:
    transport = AsyncMock(spec=DecisionTransport)
    transport.request_decision.return_value = DecodedDecision(
        StructuredData.from_json({"decision": "result", "result": "done"}),
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
            structured_data_converter=PydanticStructuredDataConverter(),
            prompt_renderer=Mock(spec=PromptRenderer),
            decision_transport=transport,
            stream=stream,
            max_repair_attempts=max_repair_attempts,
        )

    return factory
