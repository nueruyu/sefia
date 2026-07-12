import inspect
from typing import Awaitable, Callable

from sefia._interfaces.middleware import StepContext, StepMiddleware
from sefia.inference import HistoryItem, InferenceDecision, ToolCallDecision

Compactor = Callable[
    [list[HistoryItem]], list[HistoryItem] | Awaitable[list[HistoryItem]]
]


def truncate_history(history: list[HistoryItem], keep_items: int) -> list[HistoryItem]:
    """
    Keep the newest items starting at a decision boundary.

    Results whose decision was discarded are dropped as well.
    """
    if len(history) <= keep_items:
        return list(history)
    tail = list(history[-keep_items:]) if keep_items > 0 else []
    start = next(
        (i for i, item in enumerate(tail) if isinstance(item, ToolCallDecision)),
        len(tail),
    )
    return tail[start:]


class HistoryCompactor(StepMiddleware):
    """
    Compacts the run's history before a step once it grows past ``max_items``.

    The rewrite goes through ``ctx.history.rewrite``, so it is persisted before
    the model sees it. Because history is durable (the default
    ``GlyffHistoryStorage`` and ``SessionHistoryStorage`` both save each
    snapshot), a resume loads the compacted history directly, so any compactor
    is safe. Compaction changes the content subsequent steps are keyed on, so
    the first step after a rewrite is a fresh model call.

    ``compact`` receives a copy of the history and returns the replacement; it
    may be synchronous or asynchronous. The default keeps ``keep_items`` at a
    decision boundary; ``keep_items`` defaults to half of ``max_items``.
    """

    def __init__(
        self,
        *,
        max_items: int,
        keep_items: int | None = None,
        compact: Compactor | None = None,
    ):
        if max_items < 1:
            raise ValueError("max_items must be at least 1")
        if compact is not None and keep_items is not None:
            raise ValueError("Pass either keep_items or compact, not both.")
        if keep_items is not None and keep_items < 0:
            raise ValueError("keep_items must not be negative")
        self.max_items = max_items
        self._keep_items = keep_items if keep_items is not None else max_items // 2
        self._compact = compact

    async def wrap(
        self,
        ctx: StepContext,
        nxt: Callable[[], Awaitable[InferenceDecision]],
    ) -> InferenceDecision:
        if len(ctx.history.items) > self.max_items:
            compacted = await self._run_compact(list(ctx.history.items))
            await ctx.history.rewrite(compacted)
        return await nxt()

    async def _run_compact(self, history: list[HistoryItem]) -> list[HistoryItem]:
        if self._compact is None:
            return truncate_history(history, self._keep_items)
        result = self._compact(history)
        if inspect.isawaitable(result):
            result = await result
        return result
