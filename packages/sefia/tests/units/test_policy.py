from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from unittest.mock import Mock

from typing_extensions import final, override

import pytest

from sefia import (
    DecisionContext,
    DecisionMiddleware,
    InferenceMiddleware,
    Policy,
    StepMiddleware,
    policy,
)
from sefia._authoring.domain import _partition_middleware
from sefia.inference import StepDecision
from sefia.event_system import Event, EventHandler


class _Handler(EventHandler[Event]):
    async def handle(self, event: Event) -> None:
        pass


@dataclass
class _PolicyFixture(Policy):
    count: int


def test_policy_contributes_nothing_by_default():
    p = Policy()

    assert p.create_handlers() == []
    assert p.create_middleware() == []


def test_policy_calls_factories_once_per_create():
    built: list[_Handler] = []

    def make_handlers() -> list[EventHandler[Event]]:
        handler = _Handler()
        built.append(handler)
        return [handler]

    p = Policy(handlers=make_handlers)

    first = p.create_handlers()
    second = p.create_handlers()

    # Each create call goes through the factory, so per-run state is fresh.
    assert built == [*first, *second]
    assert first[0] is not second[0]


def test_dataclass_subclass_need_not_call_init():
    @dataclass
    class _MiddlewareOnly(Policy):
        label: str

        def create_middleware(
            self,
        ) -> list[InferenceMiddleware | StepMiddleware | DecisionMiddleware]:
            return []

    p = _MiddlewareOnly(label="x")

    # The non-overridden method falls back to the class-level empty default.
    assert p.create_handlers() == []


def test_policy_decorator_rejects_non_policy() -> None:
    with pytest.raises(TypeError):
        policy(_PolicyFixture)  # type: ignore


@final
class _DecisionMiddleware(DecisionMiddleware):
    @override
    async def wrap(
        self, ctx: DecisionContext, nxt: Callable[[], Awaitable[StepDecision]]
    ) -> StepDecision:
        return await nxt()


@final
class _DecisionPolicy(Policy):
    @override
    def create_middleware(
        self,
    ) -> list[InferenceMiddleware | StepMiddleware | DecisionMiddleware]:
        return [_DecisionMiddleware()]


@pytest.mark.parametrize(
    "p", [Policy(middleware=lambda: [_DecisionMiddleware()]), _DecisionPolicy()]
)
def test_policy_supports_fresh_decision_middleware(p: Policy) -> None:
    first = p.create_middleware()
    second = p.create_middleware()
    assert len(first) == len(second) == 1
    assert isinstance(first[0], DecisionMiddleware)
    assert isinstance(second[0], DecisionMiddleware)
    assert first[0] is not second[0]


def test_partition_preserves_order_within_each_middleware_scope() -> None:
    inference = [Mock(spec=InferenceMiddleware), Mock(spec=InferenceMiddleware)]
    step = [Mock(spec=StepMiddleware), Mock(spec=StepMiddleware)]
    decision = [_DecisionMiddleware(), _DecisionMiddleware()]

    assert _partition_middleware(
        [decision[0], step[0], inference[0], step[1], inference[1], decision[1]]
    ) == (inference, step, decision)


def test_partition_rejects_unsupported_middleware() -> None:
    with pytest.raises(
        TypeError,
        match="InferenceMiddleware, StepMiddleware, or DecisionMiddleware, got object",
    ):
        _partition_middleware([object()])
