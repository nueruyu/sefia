import glyff
import sefia
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol

from sefia._executor import InferenceExecutor
from sefia._interfaces import InferenceStrategy
from sefia.event_system import EventPublisher
from sefia.inference import FunctionInfo, HistoryItem, ResultDecision, StepDecision
from sefia.tool_collectors import DefaultToolCollector

from sefia.testing import MemoryHistoryStorage

infer = sefia.Domain(
    glyff.Domain(
        "packages.sefia.tests.integrations.test_executor_tool_surface", version="1"
    )
).infer


class _StubStrategy(InferenceStrategy):
    def __init__(self) -> None:
        self.tool_names: list[str] | None = None

    async def decide_next_step(
        self,
        function_info: FunctionInfo,
        history: Sequence[HistoryItem],
        tools: sefia.ToolRegistry,
        publisher: EventPublisher,
    ) -> StepDecision:
        self.tool_names = tools.get_names()
        return ResultDecision(result="done")


class BothSurface(Protocol):
    """A surface shared by two @infer methods, declaring both."""

    def run(self, topic: str) -> Awaitable[str]: ...

    def analyze(self, topic: str) -> Awaitable[str]: ...


class Service:
    @infer
    async def run(self: BothSurface, topic: str) -> str:
        """Entry point."""
        ...

    @infer
    async def analyze(self, topic: str) -> str:
        """A sibling inference, exposed as a tool via the surface."""
        ...


def _executor_for(
    bound_wrapper: Callable[..., Any], strategy: InferenceStrategy, *args: Any
) -> InferenceExecutor:
    return InferenceExecutor(
        func=inspect.unwrap(bound_wrapper),
        args=args,
        kwargs={},
        inference_strategy=strategy,
        tool_collector=DefaultToolCollector(),
        engrave=lambda _name, f: f,
        publisher=EventPublisher([]),
        history_storage=MemoryHistoryStorage(),
    )


async def test_a_surface_grants_exactly_what_it_declares_including_the_running_method():
    # A surface is an explicit allowlist with no hidden exceptions: declaring
    # the running @infer method exposes it to itself. Recursion is therefore a
    # declared choice; bounding it is runtime policy, not discovery.
    strategy = _StubStrategy()
    executor = _executor_for(Service.run, strategy, Service(), "topic")

    assert await executor.run() == "done"
    assert sorted(strategy.tool_names or []) == [
        "BothSurface_analyze",
        "BothSurface_run",
    ]
