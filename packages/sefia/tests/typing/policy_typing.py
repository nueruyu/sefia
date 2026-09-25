from typing import assert_type

from sefia import (
    DecisionMiddleware,
    InferenceMiddleware,
    MiddlewareSet,
    Policy,
    StepMiddleware,
    policy,
)
from typing_extensions import override


policy(Policy())
policy("not a policy")  # pyright: ignore[reportArgumentType]


class StepPolicy(Policy):
    @override
    def create_middleware(self) -> MiddlewareSet:
        return MiddlewareSet()


class DecisionPolicy(Policy):
    @override
    def create_middleware(self) -> MiddlewareSet:
        return MiddlewareSet()


class InferencePolicy(Policy):
    @override
    def create_middleware(self) -> MiddlewareSet:
        return MiddlewareSet()


def check_policy_factories(
    steps: list[StepMiddleware],
    decisions: tuple[DecisionMiddleware, ...],
    inference: list[InferenceMiddleware],
) -> None:
    assert_type(
        Policy(middleware=lambda: MiddlewareSet(step=tuple(steps))).create_middleware(),
        MiddlewareSet,
    )
    assert_type(
        Policy(
            middleware=lambda: MiddlewareSet(decision=decisions)
        ).create_middleware(),
        MiddlewareSet,
    )
    assert_type(
        Policy(
            middleware=lambda: MiddlewareSet(inference=tuple(inference))
        ).create_middleware(),
        MiddlewareSet,
    )
    Policy(middleware=lambda: [object()])  # pyright: ignore[reportArgumentType]
