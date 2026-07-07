"""Shared fixtures for the sefios test tree.

This tree is not a package (no ``__init__.py``), so shared helpers are exposed
as fixtures rather than importable symbols.
"""

from typing import Any, Callable, Coroutine

import pytest
from glyff import ArgsHasher, Serializer
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer
from sefia.llm import LLMClient, LLMResponse, Message


class MockLLMClient(LLMClient):
    """A mock LLM client that returns pre-defined responses."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        output_schema: dict | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
    ) -> LLMResponse:
        self.requests.append(
            {
                "messages": [m.to_dict(exclude_none=True) for m in messages],
                "tools": tools,
                "output_schema": output_schema,
                "stream_callback": stream_callback,
            }
        )
        if not self.responses:
            raise AssertionError("MockLLMClient has no more responses.")
        return self.responses.pop(0)


@pytest.fixture
def make_mock_llm() -> Callable[[list[LLMResponse]], MockLLMClient]:
    """Factory for the shared mock LLM client."""

    def factory(responses: list[LLMResponse]) -> MockLLMClient:
        return MockLLMClient(responses=responses)

    return factory


@pytest.fixture
def serializer() -> Serializer:
    return PydanticSerializer()


@pytest.fixture
def hasher() -> ArgsHasher:
    return PydanticArgsHasher()
