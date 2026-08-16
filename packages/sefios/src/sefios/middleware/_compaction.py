from typing import Awaitable, Callable

from sefia._interfaces.middleware import StepContext, StepMiddleware
from sefia.inference import HistoryItem, StepDecision, ToolCallsDecision
from typing_extensions import final, override


def _truncate_history(history: list[HistoryItem], keep_items: int) -> list[HistoryItem]:
    """
    Keep the newest items starting at a decision boundary.

    Results whose decision was discarded are dropped as well.
    """
    if len(history) <= keep_items:
        return list(history)
    tail = list(history[-keep_items:]) if keep_items > 0 else []
    start = next(
        (i for i, item in enumerate(tail) if isinstance(item, ToolCallsDecision)),
        len(tail),
    )
    return tail[start:]


@final
class HistoryCompactor(StepMiddleware):
    """Truncates history at a decision boundary once it exceeds ``max_items``."""

    def __init__(
        self,
        *,
        max_items: int,
        keep_items: int | None = None,
    ):
        if max_items < 1:
            raise ValueError("max_items must be at least 1")
        if keep_items is not None and keep_items < 0:
            raise ValueError("keep_items must not be negative")
        self.max_items = max_items
        self._keep_items = keep_items if keep_items is not None else max_items // 2

    @override
    async def wrap(
        self,
        ctx: StepContext,
        nxt: Callable[[], Awaitable[StepDecision]],
    ) -> StepDecision:
        if len(ctx.history.items) > self.max_items:
            compacted = _truncate_history(list(ctx.history.items), self._keep_items)
            ctx.history.rewrite(compacted)
        return await nxt()
