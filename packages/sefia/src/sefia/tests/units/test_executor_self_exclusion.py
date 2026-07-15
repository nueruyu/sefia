import inspect
from typing import Protocol

from sefia import infer
from sefia._executor import InferenceExecutor
from sefia._interfaces import InferenceStrategy
from sefia.event_system import EventPublisher
from sefia.tool_collectors import DefaultToolCollector


class _StubStrategy(InferenceStrategy):
    async def decide_next_step(self, function_info, history, tools):
        raise AssertionError("not driven in this test")


class BothSurface(Protocol):
    """A surface that (incorrectly) declares the running method itself."""

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
        engrave=lambda f: f,
        publisher=EventPublisher([]),
    )


def test_the_running_infer_method_is_excluded_from_its_own_tools():
    service = Service()
    executor = _executor_for(Service.run, service, "topic")

    names = executor._tool_registry.get_names()
    # The sibling declared by the surface is a tool; the method itself is not.
    assert names == ["BothSurface_analyze"]
