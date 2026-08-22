from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .._history import StepHistory
from .._tool_system import ToolRegistry
from ..inference import StepDecision


@dataclass
class InferenceContext:
    """
    Context handed to an :class:`InferenceMiddleware` wrapping a whole inference
    run (one execution of the step loop).
    """

    func_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass
class StepContext:
    """
    Context handed to a :class:`StepMiddleware` wrapping a single inference step
    (one call to the inference strategy).

    ``step`` is the 0-based index of the step about to run. ``history.items`` is
    immutable; middleware may reshape the history via ``history.rewrite``.
    """

    step: int
    history: StepHistory
    tool_registry: ToolRegistry = field(default_factory=ToolRegistry)


class InferenceMiddleware(ABC):
    """
    Wraps a full inference run.

    Unlike an :class:`~sefia.EventHandler` (which observes), a middleware
    *controls*: it may run the wrapped inference, inspect the outcome, retry by
    calling ``nxt`` again, or raise an exception to stop the run.
    """

    @abstractmethod
    async def wrap(
        self, ctx: InferenceContext, nxt: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Run the wrapped inference (via ``nxt``) and return its result."""
        ...


class StepMiddleware(ABC):
    """
    Wraps a single inference step (one inference-strategy decision).

    The executor owns the step loop and invokes the middleware once per step.
    A middleware may short-circuit the step (e.g. refuse to start it) or inspect
    the resulting decision, raising an exception to stop the loop.
    """

    @abstractmethod
    async def wrap(
        self,
        ctx: StepContext,
        nxt: Callable[[], Awaitable[StepDecision]],
    ) -> StepDecision:
        """Run the wrapped step (via ``nxt``) and return its decision."""
        ...
