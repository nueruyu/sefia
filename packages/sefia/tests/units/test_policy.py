from dataclasses import dataclass

import pytest

from sefia import Policy, policy
from sefia._interfaces import InferenceMiddleware, StepMiddleware
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

        def create_middleware(self) -> list[InferenceMiddleware | StepMiddleware]:
            return []

    p = _MiddlewareOnly(label="x")

    # The non-overridden method falls back to the class-level empty default.
    assert p.create_handlers() == []


def test_policy_decorator_rejects_non_policy() -> None:
    with pytest.raises(TypeError):
        policy(_PolicyFixture)  # type: ignore
