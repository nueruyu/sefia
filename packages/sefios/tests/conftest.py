"""Shared fixtures for the sefios test tree.

This tree is not a package (no ``__init__.py``), so shared helpers are exposed
as fixtures rather than importable symbols; the test doubles themselves live in
the public ``sefia.testing`` module.
"""

from typing import Callable

import pytest
from glyff import ArgumentCanonicalizer, Serializer
from glyff.serialization import FallbackByTypeQualname
from glyff_pydantic import PydanticArgumentCanonicalizer, PydanticSerializer
from sefia.llm import LLMCompletion
from sefia.testing import MockLLMClient
from sefios.storage import MemorySessionStorage


@pytest.fixture
def make_mock_llm() -> Callable[[list[LLMCompletion]], MockLLMClient]:
    """Factory for the shared mock LLM client."""

    def factory(completions: list[LLMCompletion]) -> MockLLMClient:
        return MockLLMClient(completions=completions)

    return factory


@pytest.fixture
def serializer() -> Serializer:
    return PydanticSerializer()


@pytest.fixture
def memory_session_storage(serializer: Serializer) -> MemorySessionStorage:
    return MemorySessionStorage(serializer)


@pytest.fixture
def hasher() -> ArgumentCanonicalizer:
    return PydanticArgumentCanonicalizer(FallbackByTypeQualname())
