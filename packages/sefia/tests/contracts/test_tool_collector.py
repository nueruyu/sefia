"""Apply the public tool-collector contract to every built-in collector."""

import inspect

import pytest

import sefia.tool_collectors as collectors
from sefia import JsonSchemaToolEntry, ToolCollector, Tools
from sefia.inference import Capability
from sefia.testing import ToolCollectorCase, ToolCollectorContract
from sefia.tool_collectors import (
    CompositeToolCollector,
    DefaultToolCollector,
    StaticToolCollector,
)

COLLECTOR_TYPES = (
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


def _static_collector() -> StaticToolCollector:
    async def lookup() -> str:
        return "ok"

    return StaticToolCollector(
        [
            JsonSchemaToolEntry(
                lookup,
                name="lookup",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )
        ]
    )


class TestDefaultToolCollectorContract(ToolCollectorContract):
    @pytest.fixture
    def tool_collector_case(self) -> ToolCollectorCase:
        return ToolCollectorCase(
            DefaultToolCollector(),
            [Capability(_Agent(), _Agent)],
            "_Toolkit_lookup",
            expected_result="ok",
        )


class TestStaticToolCollectorContract(ToolCollectorContract):
    @pytest.fixture
    def tool_collector_case(self) -> ToolCollectorCase:
        return ToolCollectorCase(
            _static_collector(), [], "lookup", expected_result="ok"
        )


class TestCompositeToolCollectorContract(ToolCollectorContract):
    @pytest.fixture
    def tool_collector_case(self) -> ToolCollectorCase:
        return ToolCollectorCase(
            CompositeToolCollector([_static_collector()]),
            [],
            "lookup",
            expected_result="ok",
        )


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in collectors.__all__
        if inspect.isclass(value := getattr(collectors, name))
        and issubclass(value, ToolCollector)
    }

    assert set(COLLECTOR_TYPES) == exported
