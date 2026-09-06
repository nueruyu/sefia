from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock

from sefia import (
    HistorySnapshot,
    StepContext,
    StepMiddleware,
    ToolRegistry,
)
from sefia._executor import InferenceExecutor
from sefia.events import StepStarted
from sefia.inference import (
    ResultDecision,
    StepDecision,
    ToolCallResult,
    ToolCallsDecision,
)
from sefia.testing import MemoryHistoryStorage, make_tool_call_request
from typing_extensions import final, override


async def test_resumes_loop_from_stored_snapshot(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    mock_publisher: AsyncMock,
) -> None:
    stored_items = (
        ToolCallsDecision(
            calls=[make_tool_call_request(id="1", name="a_tool", arguments={})]
        ),
        ToolCallResult(tool_call_id="1", result="earlier"),
    )
    # completed_steps deliberately larger than the (compacted) item count.
    snapshot = HistorySnapshot(items=stored_items, completed_steps=4)
    mock_strategy.decide_next_step.return_value = ResultDecision(result="done")

    executor = make_executor(
        history_storage=MemoryHistoryStorage(snapshot),
    )

    result = await executor.run()

    assert result == "done"
    history = mock_strategy.decide_next_step.call_args.kwargs["history"]
    assert list(history) == list(stored_items)
    step_events = [
        call.args[0]
        for call in mock_publisher.publish.call_args_list
        if isinstance(call.args[0], StepStarted)
    ]
    assert [event.step for event in step_events] == [4]


async def test_saves_snapshot_after_each_completed_step(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    mock_collector: MagicMock,
) -> None:
    decision = ToolCallsDecision(
        calls=[make_tool_call_request(id="1", name="my_tool", arguments={"a": 1})]
    )
    empty_decision = ToolCallsDecision(calls=[])
    mock_strategy.decide_next_step.side_effect = [
        decision,
        empty_decision,
        ResultDecision(result="final"),
    ]

    tool_registry = ToolRegistry()
    tool_registry.add(AsyncMock(return_value="tool result"), name="my_tool")
    mock_collector.collect.return_value = tool_registry

    storage = MemoryHistoryStorage()
    executor = make_executor(
        history_storage=storage,
    )

    await executor.run()

    first = (decision, ToolCallResult(tool_call_id="1", result="tool result"))
    assert [(s.items, s.completed_steps) for s in storage.saves] == [
        (first, 1),
        ((*first, empty_decision), 2),
    ]


async def test_compaction_is_persisted_before_the_model_call(
    make_executor: Callable[..., InferenceExecutor], mock_strategy: AsyncMock
) -> None:
    seeded = (
        ToolCallResult(tool_call_id="0", result="old"),
        ToolCallResult(tool_call_id="1", result="recent"),
    )
    storage = MemoryHistoryStorage(HistorySnapshot(items=seeded, completed_steps=2))

    @final
    class _KeepLast(StepMiddleware):
        @override
        async def wrap(
            self,
            ctx: StepContext,
            nxt: Callable[[], Awaitable[StepDecision]],
        ) -> StepDecision:
            ctx.history.rewrite([ctx.history.items[-1]])
            return await nxt()

    async def decide(**kwargs: object) -> ResultDecision:
        assert [(s.items, s.completed_steps) for s in storage.saves] == [
            ((seeded[-1],), 2)
        ]
        return ResultDecision(result="done")

    mock_strategy.decide_next_step.side_effect = decide

    executor = make_executor(
        history_storage=storage,
        step_middlewares=[_KeepLast()],
    )

    await executor.run()

    # The compacted history was persisted (step count unchanged), and the
    # model was called with it.
    assert [(s.items, s.completed_steps) for s in storage.saves] == [
        ((seeded[-1],), 2),
    ]
    history = mock_strategy.decide_next_step.call_args.kwargs["history"]
    assert list(history) == [seeded[-1]]
