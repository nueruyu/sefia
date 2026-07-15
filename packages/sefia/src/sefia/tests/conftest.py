"""Shared fixtures for the sefia test tree.

Shared helpers live in the public ``sefia.testing`` module rather than here;
this file only provides fixtures. Domain doubles (toolkits, agents, report
types) are defined locally in the test files that use them.
"""

import pytest
from glyff import ArgsHasher, Serializer
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer


@pytest.fixture
def serializer() -> Serializer:
    return PydanticSerializer()


@pytest.fixture
def hasher() -> ArgsHasher:
    return PydanticArgsHasher()
