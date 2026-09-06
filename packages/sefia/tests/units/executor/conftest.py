from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sefia import (
    HistoryStorage,
    InferenceContext,
    InferenceMiddleware,
    InferenceStrategy,
    ToolCollector,
    ToolRegistry,
)
from sefia._executor import InferenceExecutor
from sefia.event_system import EventPublisher
from sefia.testing import MemoryHistoryStorage
from typing_extensions import final, override


@pytest.fixture
def mock_strategy() -> AsyncMock:
    return AsyncMock(spec=InferenceStrategy)


@pytest.fixture
def mock_collector() -> MagicMock:
    collector = MagicMock(spec=ToolCollector)
    collector.collect.return_value = ToolRegistry()
    return collector


@pytest.fixture
def mock_publisher() -> AsyncMock:
    return AsyncMock(spec=EventPublisher)


@pytest.fixture
def make_executor(
    mock_strategy: AsyncMock, mock_collector: MagicMock, mock_publisher: AsyncMock
) -> Callable[..., InferenceExecutor]:
    def task(value: str) -> str:
        return value

    def identity_engrave(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        return function

    def factory(
        *, history_storage: HistoryStorage | None = None, **kwargs: Any
    ) -> InferenceExecutor:
        engrave = kwargs.pop("engrave", identity_engrave)
        return InferenceExecutor(
            task,
            ("value",),
            {},
            mock_strategy,
            mock_collector,
            engrave,
            mock_publisher,
            history_storage=history_storage
            if history_storage is not None
            else MemoryHistoryStorage(),
            **kwargs,
        )

    return factory


@pytest.fixture
def retry_once() -> InferenceMiddleware:
    @final
    class RetryOnce(InferenceMiddleware):
        @override
        async def wrap(
            self, ctx: InferenceContext, nxt: Callable[[], Awaitable[Any]]
        ) -> Any:
            try:
                return await nxt()
            except ValueError:
                return await nxt()

    return RetryOnce()
