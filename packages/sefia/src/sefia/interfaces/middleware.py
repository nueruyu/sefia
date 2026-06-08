from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..models import HistoryItem, InferenceDecision


@dataclass
class RunContext:
    """
    Context handed to an :class:`InferenceMiddleware` wrapping a whole inference
    run (one execution of the step loop).
    """

    func_name: str
    args: tuple
    kwargs: dict


@dataclass
class StepContext:
    """
    Context handed to a :class:`StepMiddleware` wrapping a single inference step
    (one call to the inference strategy).

    ``step`` is the 0-based index of the step about to run.
    """

    step: int
    history: list[HistoryItem]


class InferenceMiddleware(ABC):
    """
    Wraps a full inference run.

    Unlike an :class:`~sefia.interfaces.EventHandler` (which observes), a
    middleware *controls*: it may run the wrapped inference, inspect the
    outcome, and steer the executor's retry loop by raising a typed control
    signal (see ``sefia.middleware.signals``). The executor owns the loop;
    middleware never loops on its own, so multiple middlewares compose cleanly.
    """

    @abstractmethod
    async def wrap(self, ctx: RunContext, nxt: Callable[[], Awaitable[Any]]) -> Any:
        """Run the wrapped inference (via ``nxt``) and return its result."""
        ...


class StepMiddleware(ABC):
    """
    Wraps a single inference step (one inference-strategy decision).

    The executor owns the step loop and invokes the middleware once per step.
    A middleware may short-circuit the step (e.g. refuse to start it) or inspect
    the resulting decision, raising a typed control signal to stop the loop.
    """

    @abstractmethod
    async def wrap(
        self,
        ctx: StepContext,
        nxt: Callable[[], Awaitable[InferenceDecision]],
    ) -> InferenceDecision:
        """Run the wrapped step (via ``nxt``) and return its decision."""
        ...
