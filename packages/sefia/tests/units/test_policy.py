from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from typing_extensions import final, override

import pytest

from sefia import (
    DecisionContext,
    DecisionMiddleware,
    MiddlewareSet,
    Policy,
    policy,
)
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
    assert p.create_middleware() == MiddlewareSet()


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

        def create_middleware(self) -> MiddlewareSet:
            return MiddlewareSet()

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
    def create_middleware(self) -> MiddlewareSet:
        return MiddlewareSet(decision=(_DecisionMiddleware(),))


@pytest.mark.parametrize(
    "p",
    [
        Policy(middleware=lambda: MiddlewareSet(decision=(_DecisionMiddleware(),))),
        _DecisionPolicy(),
    ],
)
def test_policy_supports_fresh_decision_middleware(p: Policy) -> None:
    first = p.create_middleware()
    second = p.create_middleware()
    assert first is not second
    assert len(first.decision) == len(second.decision) == 1
    assert isinstance(first.decision[0], DecisionMiddleware)
    assert isinstance(second.decision[0], DecisionMiddleware)
    assert first.decision[0] is not second.decision[0]


def test_middleware_set_preserves_category_order() -> None:
    first, second = _DecisionMiddleware(), _DecisionMiddleware()
    middleware = MiddlewareSet(decision=(first, second))

    assert middleware.decision == (first, second)
