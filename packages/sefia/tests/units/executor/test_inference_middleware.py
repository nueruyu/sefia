from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest
from sefia import InferenceMiddleware
from sefia._executor import InferenceExecutor
from sefia.inference import ResultDecision


async def test_inference_middleware_retries_inference_failure(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    retry_once: InferenceMiddleware,
) -> None:
    mock_strategy.decide_next_step.side_effect = [
        ValueError("flaky inference"),
        ResultDecision(result="second attempt"),
    ]

    executor = make_executor(
        inference_middlewares=[retry_once],
    )

    result = await executor.run()

    assert result == "second attempt"
    assert mock_strategy.decide_next_step.call_count == 2


async def test_failure_from_reentered_attempt_propagates(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    retry_once: InferenceMiddleware,
) -> None:
    mock_strategy.decide_next_step.side_effect = ValueError("always flaky")

    executor = make_executor(
        inference_middlewares=[retry_once],
    )

    with pytest.raises(ValueError, match="always flaky"):
        await executor.run()

    # The second failure propagates out of middleware.
    assert mock_strategy.decide_next_step.call_count == 2
