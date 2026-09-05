"""Shared behavior required of every tool collector."""

import inspect
from dataclasses import dataclass
from typing import TypeAlias

import pytest

import sefia.tool_collectors as collectors
from sefia import JsonSchemaToolEntry, ToolCollector, Tools
from sefia.inference import Capability
from sefia.tool_collectors import (
    CompositeToolCollector,
    DefaultToolCollector,
    StaticToolCollector,
)


CollectorType: TypeAlias = (
    type[CompositeToolCollector]
    | type[DefaultToolCollector]
    | type[StaticToolCollector]
)
COLLECTOR_TYPES: tuple[CollectorType, ...] = (
    CompositeToolCollector,
    DefaultToolCollector,
    StaticToolCollector,
)


class _Toolkit:
    async def lookup(self) -> str:
        return "ok"


class _Agent:
    toolkit: Tools[_Toolkit]

    def __init__(self) -> None:
        self.toolkit = _Toolkit()


@dataclass(frozen=True)
class _CollectorCase:
    collector: ToolCollector
    capabilities: list[Capability]


def _static_tool() -> JsonSchemaToolEntry:
    async def lookup() -> str:
        return "ok"

    return JsonSchemaToolEntry(
        lookup,
        name="lookup",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )


@pytest.fixture(
    params=COLLECTOR_TYPES, ids=lambda collector_type: collector_type.__name__
)
def collector_case(request: pytest.FixtureRequest) -> _CollectorCase:
    collector_type = request.param
    if collector_type is DefaultToolCollector:
        return _CollectorCase(collector_type(), [Capability(_Agent(), _Agent)])
    static = StaticToolCollector([_static_tool()])
    if collector_type is CompositeToolCollector:
        return _CollectorCase(collector_type([static]), [])
    return _CollectorCase(static, [])


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in collectors.__all__
        if inspect.isclass(value := getattr(collectors, name))
        and issubclass(value, ToolCollector)
    }

    assert set(COLLECTOR_TYPES) == exported


async def test_contract_collects_executable_tools(
    collector_case: _CollectorCase,
) -> None:
    registry = collector_case.collector.collect(collector_case.capabilities)

    assert len(registry.get_all()) == 1
    assert await registry.get_all()[0].invoke({}) == "ok"
