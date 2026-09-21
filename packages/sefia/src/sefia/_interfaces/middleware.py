from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, cast

from .._history import StepHistory
from .._message_plan import MessagePlan
from .._tool_system import ToolRegistry
from ..inference import FunctionInfo, StepDecision


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
    outside its durable decision execution.

    ``step`` is the 0-based index of the step about to run. ``history.items`` is
    immutable; middleware may reshape the history via ``history.rewrite``.
    """

    step: int
    history: StepHistory
    tool_registry: ToolRegistry = field(default_factory=ToolRegistry)


@dataclass(frozen=True)
class DecisionContext:
    """Context for middleware wrapping decision generation inside a durable step."""

    step: int


@dataclass(frozen=True)
class MessageContext:
    step: int
    function: FunctionInfo


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
    Wraps a single inference step outside its durable decision execution.

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


class DecisionMiddleware(ABC):
    """Wraps decision generation inside one durable inference step."""

    @abstractmethod
    async def wrap(
        self,
        ctx: DecisionContext,
        nxt: Callable[[], Awaitable[StepDecision]],
    ) -> StepDecision: ...


class MessageMiddleware(ABC):
    """Wraps application message composition inside one durable decision."""

    @abstractmethod
    async def wrap(
        self,
        ctx: MessageContext,
        nxt: Callable[[], Awaitable[MessagePlan]],
    ) -> MessagePlan: ...


@dataclass(frozen=True)
class MiddlewareSet:
    inference: tuple[InferenceMiddleware, ...] = ()
    step: tuple[StepMiddleware, ...] = ()
    decision: tuple[DecisionMiddleware, ...] = ()
    message: tuple[MessageMiddleware, ...] = ()

    def __post_init__(self) -> None:
        categories: tuple[tuple[str, type[ABC], object], ...] = (
            ("inference", InferenceMiddleware, cast(object, self.inference)),
            ("step", StepMiddleware, cast(object, self.step)),
            ("decision", DecisionMiddleware, cast(object, self.decision)),
            ("message", MessageMiddleware, cast(object, self.message)),
        )
        for name, expected, items in categories:
            if not isinstance(items, tuple):
                raise TypeError(f"MiddlewareSet.{name} must be a tuple.")
            for index, item in enumerate(cast(tuple[object, ...], items)):
                if not isinstance(item, expected):
                    raise TypeError(
                        f"MiddlewareSet.{name}[{index}] must be "
                        f"{expected.__name__}; got {type(item).__name__}."
                    )
