import inspect
from typing import Awaitable, Callable

from sefia._interfaces.middleware import StepContext, StepMiddleware
from sefia.inference import HistoryItem, InferenceDecision, ToolCallDecision

Compactor = Callable[
    [list[HistoryItem]], list[HistoryItem] | Awaitable[list[HistoryItem]]
]


def truncate_history(history: list[HistoryItem], keep_items: int) -> list[HistoryItem]:
    """
    Keep the most recent ``keep_items`` history items, aligned to a decision
    boundary: leading ``ToolCallResult``s whose ``ToolCallDecision`` was cut
    off are dropped too, so the history never starts with an orphaned result.
    The run's task itself is not part of the history — the function's
    arguments are rendered into the prompt every step — so truncation only
    discards old tool interactions.
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

    The rewrite goes through ``StepContext.rewrite_history``, so it is
    persisted to the run's history store before the model sees it. With a
    persistent store (``DurableHistoryStore``) any compactor is safe: a resume
    loads the compacted snapshot instead of rebuilding history by replay. With
    the default transient store, only a *deterministic* compactor (like the
    default truncation) is replay-safe — a resume re-derives history by replay
    and must re-produce the same rewrite, or the engraved steps keyed on the
    compacted content will not be found. Either way, compaction changes the
    content subsequent steps are keyed on, so the first step after a rewrite
    is a fresh model call.

    ``compact`` receives a copy of the history and returns the replacement; it
    may be sync or async (e.g. an LLM summarizer — persistent store only).
    When omitted, the default keeps the most recent ``keep_items`` items
    (``max_items // 2`` if unset), cut at a decision boundary via
    :func:`truncate_history`.
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
        if len(ctx.history) > self.max_items:
            compacted = await self._run_compact(list(ctx.history))
            await ctx.rewrite_history(compacted)
        return await nxt()

    async def _run_compact(self, history: list[HistoryItem]) -> list[HistoryItem]:
        if self._compact is None:
            return truncate_history(history, self._keep_items)
        result = self._compact(history)
        if inspect.isawaitable(result):
            result = await result
        return result
