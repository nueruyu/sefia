from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .._tool_system import ToolRegistry
from ..inference import HistoryItem, InferenceDecision
from .history_store import HistoryStore


@dataclass
class InferenceContext:
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

    ``step`` is the 0-based index of the step about to run. ``history`` is the
    executor's own list, shared by reference — mutate it only through
    :meth:`rewrite_history` so the run's :class:`HistoryStore` stays in sync.
    """

    step: int
    history: list[HistoryItem]
    tool_registry: ToolRegistry = field(default_factory=ToolRegistry)
    history_store: HistoryStore | None = None

    async def rewrite_history(self, items: Sequence[HistoryItem]) -> None:
        """
        Replace the run's history with ``items``, for this and later steps.

        The rewrite is persisted through the run's :class:`HistoryStore`
        *before* the in-memory list is swapped, so a crash between the two
        leaves the store ahead (harmless) rather than behind (a lost rewrite).
        Note that rewriting changes the content the subsequent engraved steps
        are keyed on: with the default transient store this is only
        replay-safe if the rewrite is deterministic, since a resume rebuilds
        history by replay and re-applies the rewrite; with a persistent store
        the saved history itself is what a resume loads, so any rewrite is
        safe.
        """
        new_items = list(items)
        if self.history_store is not None:
            await self.history_store.save(new_items)
        self.history[:] = new_items


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
        nxt: Callable[[], Awaitable[InferenceDecision]],
    ) -> InferenceDecision:
        """Run the wrapped step (via ``nxt``) and return its decision."""
        ...
