import glyff
import sefia
import inspect
from typing import Protocol

from sefia._executor import InferenceExecutor
from sefia._interfaces import InferenceStrategy
from sefia.event_system import EventPublisher
from sefia.tool_collectors import DefaultToolCollector

from sefia.testing import MemoryHistoryStorage

infer = sefia.Domain(
    glyff.Domain("packages.sefia.tests.units.test_executor_tool_surface", version="1")
).infer


class _StubStrategy(InferenceStrategy):
    async def decide_next_step(self, function_info, history, tools, publisher):
        raise AssertionError("not driven in this test")


class BothSurface(Protocol):
    """A surface shared by two @infer methods, declaring both."""

    async def run(self, topic: str) -> str: ...

    async def analyze(self, topic: str) -> str: ...


class Service:
    @infer
    async def run(self: BothSurface, topic: str) -> str:
        """Entry point."""
        ...

    @infer
    async def analyze(self, topic: str) -> str:
        """A sibling inference, exposed as a tool via the surface."""
        ...


def _executor_for(bound_wrapper, *args) -> InferenceExecutor:
    return InferenceExecutor(
        func=inspect.unwrap(bound_wrapper),
        args=args,
        kwargs={},
        inference_strategy=_StubStrategy(),
        tool_collector=DefaultToolCollector(),
        engrave=lambda _name, f: f,
        publisher=EventPublisher([]),
        history_storage=MemoryHistoryStorage(),
    )


def test_a_surface_grants_exactly_what_it_declares_including_the_running_method():
    # A surface is an explicit allowlist with no hidden exceptions: declaring
    # the running @infer method exposes it to itself. Recursion is therefore a
    # declared choice; bounding it is runtime policy, not discovery.
    executor = _executor_for(Service.run, Service(), "topic")

    assert sorted(executor._tool_registry.get_names()) == [
        "BothSurface_analyze",
        "BothSurface_run",
    ]
