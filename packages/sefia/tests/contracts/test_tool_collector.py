"""Apply the public tool-collector contract to every built-in collector."""

import inspect
from collections.abc import Callable
from typing import cast

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
from typing_extensions import override


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


CASE_FACTORIES: dict[type[ToolCollector], Callable[[], ToolCollectorCase]] = {
    DefaultToolCollector: lambda: ToolCollectorCase(
        DefaultToolCollector(),
        [Capability(_Agent(), _Agent)],
        "_Toolkit_lookup",
        expected_result="ok",
    ),
    StaticToolCollector: lambda: ToolCollectorCase(
        _static_collector(),
        [],
        "lookup",
        expected_result="ok",
    ),
    CompositeToolCollector: lambda: ToolCollectorCase(
        CompositeToolCollector([_static_collector()]),
        [],
        "lookup",
        expected_result="ok",
    ),
}


class TestToolCollectorContract(ToolCollectorContract):
    _case: ToolCollectorCase

    @pytest.fixture(
        autouse=True,
        params=tuple(CASE_FACTORIES),
        ids=[cls.__name__ for cls in CASE_FACTORIES],
    )
    def _prepare_case(self, request: pytest.FixtureRequest) -> None:
        implementation = cast(type[ToolCollector], request.param)
        self._case = CASE_FACTORIES[implementation]()

    @override
    def make_tool_collector_case(self) -> ToolCollectorCase:
        return self._case


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in collectors.__all__
        if inspect.isclass(value := getattr(collectors, name))
        and issubclass(value, ToolCollector)
    }

    assert set(CASE_FACTORIES) == exported
