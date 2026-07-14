"""Shared fixtures for the sefios test tree.

This tree is not a package (no ``__init__.py``), so shared helpers are exposed
as fixtures rather than importable symbols; the test doubles themselves live in
the public ``sefia.testing`` module.
"""

from typing import Callable

import pytest
from glyff import ArgsHasher, Serializer
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer
from sefia.llm import LLMResponse
from sefia.testing import MockLLMClient


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
