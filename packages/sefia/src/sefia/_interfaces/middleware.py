from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..inference import HistoryItem, InferenceDecision


@dataclass
class InferenceContext:
    """Context passed to middleware for a whole inference run."""

    func_name: str
    args: tuple
    kwargs: dict


@dataclass
class StepContext:
    """Context passed to middleware for one inference step."""

    step: int
    history: list[HistoryItem]


class InferenceMiddleware(ABC):
    """Wraps a whole inference run."""

    @abstractmethod
    async def wrap(self, ctx: InferenceContext, nxt: Callable[[], Awaitable[Any]]) -> Any:
        """Run the wrapped inference via ``nxt`` and return its result."""
        ...


class StepMiddleware(ABC):
    """Wraps one inference-strategy decision."""

    @abstractmethod
    async def wrap(
        self,
        ctx: StepContext,
        nxt: Callable[[], Awaitable[InferenceDecision]],
    ) -> InferenceDecision:
        """Run the wrapped step via ``nxt`` and return its decision."""
        ...
