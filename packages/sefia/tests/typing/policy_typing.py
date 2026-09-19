from collections.abc import Sequence
from typing import assert_type

from sefia import (
    DecisionMiddleware,
    InferenceMiddleware,
    Middleware,
    Policy,
    StepMiddleware,
    policy,
)
from typing_extensions import override


policy(Policy())
policy("not a policy")  # pyright: ignore[reportArgumentType]


class StepPolicy(Policy):
    @override
    def create_middleware(self) -> list[StepMiddleware]:
        return []


class DecisionPolicy(Policy):
    @override
    def create_middleware(self) -> tuple[DecisionMiddleware, ...]:
        return ()


class InferencePolicy(Policy):
    @override
    def create_middleware(self) -> list[InferenceMiddleware]:
        return []


def check_policy_factories(
    steps: list[StepMiddleware],
    decisions: tuple[DecisionMiddleware, ...],
    inference: list[InferenceMiddleware],
) -> None:
    assert_type(
        Policy(middleware=lambda: steps).create_middleware(), Sequence[Middleware]
    )
    assert_type(
        Policy(middleware=lambda: decisions).create_middleware(), Sequence[Middleware]
    )
    assert_type(
        Policy(middleware=lambda: inference).create_middleware(), Sequence[Middleware]
    )
    Policy(middleware=lambda: [object()])  # pyright: ignore[reportArgumentType]
