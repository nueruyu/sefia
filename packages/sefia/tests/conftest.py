"""Shared fixtures for the sefia test tree.

Shared helpers live in the public ``sefia.testing`` module rather than here;
this file only provides fixtures. Domain doubles (toolkits, agents, report
types) are defined locally in the test files that use them.
"""

import pytest
from glyff import ArgumentCanonicalizer, Serializer
from glyff.serialization import FallbackByTypeQualname
from glyff_pydantic import PydanticArgumentCanonicalizer, PydanticSerializer


@pytest.fixture
def serializer() -> Serializer:
    return PydanticSerializer()


@pytest.fixture
def hasher() -> ArgumentCanonicalizer:
    return PydanticArgumentCanonicalizer(FallbackByTypeQualname())
