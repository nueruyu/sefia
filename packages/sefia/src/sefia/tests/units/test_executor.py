import asyncio
from unittest.mock import AsyncMock

import pytest
from sefia.exceptions import PauseException
from pytest_mock import MockerFixture

from sefia import (
    InferenceContext,
    InferenceMiddleware,
    InferenceStrategy,
    StepContext,
    StepMiddleware,
    ToolCollector,
    ToolRegistry,
    events,
)
from sefia._executor import InferenceExecutor
from sefia.event_system import EventHandler, EventPublisher
from sefia.events import AttemptStart, StepStarted
from sefia.inference import (
    ResultDecision,
    InferenceDecision,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)


class _MaxStepsExceededError(Exception):
    pass


class _MaxRetriesExceededError(Exception):
    pass


class _StepLimiter(StepMiddleware):
    def __init__(self, max_steps: int):
        self.max_steps = max_steps

    async def wrap(self, ctx: StepContext, nxt) -> InferenceDecision:
        if ctx.step >= self.max_steps:
            raise _MaxStepsExceededError()
        return await nxt()


class _Retrier(InferenceMiddleware):
    def __init__(self, max_retries: int):
        self.max_retries = max_retries
        self._retries_used = 0

    async def wrap(self, ctx: InferenceContext, nxt):
        while True:
            try:
                return await nxt()
            except (_MaxRetriesExceededError, _MaxStepsExceededError):
                raise
            except Exception as e:
                if self._retries_used >= self.max_retries:
                    raise _MaxRetriesExceededError() from e
                self._retries_used += 1


def sample_func(arg1: str) -> str:
    """Sample docstring."""
    return "implemented"


def sample_func_with_self(self, arg1: str) -> str:
    """Sample docstring."""
    return "implemented"


@pytest.fixture
def executor_dependencies(mocker: MockerFixture):
    """Provides a tuple of mocked dependencies for InferenceExecutor."""
    mock_strategy = mocker.AsyncMock(spec=InferenceStrategy)
    mock_collector = mocker.MagicMock(spec=ToolCollector)
    mock_publisher = mocker.AsyncMock(spec=EventPublisher)

    mock_collector.collect.return_value = ToolRegistry()

    def non_engrave(f):
        return f

    return (
        mock_strategy,
        mock_collector,
        mock_publisher,
        non_engrave,
    )


class TestInferenceExecutor:
    async def test_run_loop_with_tool_call_and_result(self, executor_dependencies):
        # Arrange
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies

        mock_strategy.decide_next_step.side_effect = [
            ToolCallDecision(
                calls=[ToolCallRequest(id="1", name="my_tool", arguments={"a": 1})]
            ),
            ResultDecision(result="final result"),
        ]

        tool_registry = ToolRegistry()
        mock_tool_func = AsyncMock(return_value="tool result")
        tool_registry.add(mock_tool_func, name="my_tool")
        mock_collector.collect.return_value = tool_registry

        executor = InferenceExecutor(
            sample_func_with_self,
            (object(), "value"),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        # Act
        result = await executor.run()

        # Assert
        assert result == "final result"
        assert mock_strategy.decide_next_step.call_count == 2
        mock_tool_func.assert_called_once_with(a=1)

    async def test_handles_nonexistent_tool_call(self, executor_dependencies):
        # Arrange
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.side_effect = [
            ToolCallDecision(
                calls=[ToolCallRequest(id="1", name="nonexistent_tool", arguments={})]
            ),
            ResultDecision(result="recovered"),
        ]

        executor = InferenceExecutor(
            sample_func_with_self,
            (object(), "dummy_arg"),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        # Act
        result = await executor.run()

        # Assert
        assert result == "recovered"
        history = mock_strategy.decide_next_step.call_args_list[1].kwargs["history"]
        assert len(history) == 2
        assert isinstance(history[0], ToolCallDecision)
        assert isinstance(history[1], ToolCallResult)
        assert "Error: Tool 'nonexistent_tool' not found" in history[1].result

    async def test_fires_step_started_for_every_step(self, executor_dependencies):
        # The executor fires StepStarted (1-based) at the start of each step,
        # including the first, so a handler can observe the step count.
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.side_effect = [
            ToolCallDecision(calls=[]),
            ResultDecision(result="done"),
        ]

        executor = InferenceExecutor(
            sample_func,
            ("dummy_arg",),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        await executor.run()

        step_events = [
            call.args[0]
            for call in mock_publisher.publish.call_args_list
            if isinstance(call.args[0], StepStarted)
        ]
        assert [event.step for event in step_events] == [0, 1]

    async def test_step_middleware_can_stop_the_loop(self, executor_dependencies):
        # A step middleware that raises an exception stops the loop. Here
        # StepLimiter refuses to start a fourth step (0-based index 3).
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.return_value = ToolCallDecision(calls=[])

        executor = InferenceExecutor(
            sample_func,
            ("dummy_arg",),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
            step_middlewares=[_StepLimiter(max_steps=3)],
        )

        with pytest.raises(_MaxStepsExceededError):
            await executor.run()

        assert mock_strategy.decide_next_step.call_count == 3

    async def test_step_middlewares_compose_in_declared_order(
        self, executor_dependencies
    ):
        # Multiple step middlewares wrap each step as an onion: the first in the
        # list is the outermost layer. They run once per step, not multiplied by
        # one another, and observe the steps in order.
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.side_effect = [
            ToolCallDecision(calls=[]),
            ResultDecision(result="done"),
        ]

        calls: list[str] = []

        class Recorder(StepMiddleware):
            def __init__(self, label: str):
                self.label = label

            async def wrap(self, ctx: StepContext, nxt) -> InferenceDecision:
                calls.append(f"{self.label}:enter:{ctx.step}")
                decision = await nxt()
                calls.append(f"{self.label}:exit:{ctx.step}")
                return decision

        executor = InferenceExecutor(
            sample_func,
            ("dummy_arg",),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
            step_middlewares=[Recorder("outer"), Recorder("inner")],
        )

        result = await executor.run()

        assert result == "done"
        # Two steps, each wrapped once by each middleware, outer enclosing inner.
        assert calls == [
            "outer:enter:0",
            "inner:enter:0",
            "inner:exit:0",
            "outer:exit:0",
            "outer:enter:1",
            "inner:enter:1",
            "inner:exit:1",
            "outer:exit:1",
        ]

    async def test_tool_failure_is_not_retried_but_fed_back(
        self, executor_dependencies
    ):
        # A failing tool must not trigger a retry of the whole inference. The
        # error is stringified into the history and fed back to the model, which
        # then recovers — the run never restarts (decide is called twice).
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.side_effect = [
            ToolCallDecision(
                calls=[ToolCallRequest(id="1", name="boom_tool", arguments={})]
            ),
            ResultDecision(result="recovered"),
        ]

        tool_registry = ToolRegistry()
        failing_tool = AsyncMock(side_effect=ValueError("kaboom"))
        tool_registry.add(failing_tool, name="boom_tool")
        mock_collector.collect.return_value = tool_registry

        executor = InferenceExecutor(
            sample_func_with_self,
            (object(), "value"),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
            inference_middlewares=[_Retrier(max_retries=3)],
        )

        result = await executor.run()

        assert result == "recovered"
        # No retry: the strategy was consulted exactly twice (tool call, then
        # the result informed by the error).
        assert mock_strategy.decide_next_step.call_count == 2
        history = mock_strategy.decide_next_step.call_args_list[1].kwargs["history"]
        assert isinstance(history[-1], ToolCallResult)
        assert "Error executing tool 'boom_tool'" in history[-1].result
        assert "ValueError(kaboom)" in history[-1].result

    async def test_inference_middleware_retries_inference_failure(
        self, executor_dependencies
    ):
        # An inference-call failure is retried by Retrier. The first
        # attempt fails inside decide_next_step; the second succeeds.
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.side_effect = [
            ValueError("flaky inference"),
            ResultDecision(result="second attempt"),
        ]

        executor = InferenceExecutor(
            sample_func,
            ("dummy_arg",),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
            inference_middlewares=[_Retrier(max_retries=3)],
        )

        result = await executor.run()

        assert result == "second attempt"
        assert mock_strategy.decide_next_step.call_count == 2

    async def test_inference_middleware_raises_after_exhausting_retries(
        self, executor_dependencies
    ):
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.side_effect = ValueError("always flaky")

        executor = InferenceExecutor(
            sample_func,
            ("dummy_arg",),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
            inference_middlewares=[_Retrier(max_retries=2)],
        )

        with pytest.raises(_MaxRetriesExceededError):
            await executor.run()

        # initial attempt + 2 retries
        assert mock_strategy.decide_next_step.call_count == 3

    async def test_failed_step_publishes_event_and_reraises_by_default(
        self, executor_dependencies
    ):
        # When a step fails and no handler interrupts, the original exception
        # must propagate (so glyff engraves the genuine failure), but only after
        # an InferenceStepFailed event carrying the error has been published.
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        error = ValueError("boom")
        mock_strategy.decide_next_step.side_effect = error

        executor = InferenceExecutor(
            sample_func,
            ("dummy_arg",),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        with pytest.raises(ValueError, match="boom"):
            await executor.run()

        published = [call.args[0] for call in mock_publisher.publish.call_args_list]
        step_failures = [
            e for e in published if isinstance(e, events.InferenceStepFailed)
        ]
        assert len(step_failures) == 1
        assert step_failures[0].error is error

    async def test_recoverable_inference_error_yields_without_failing_run(
        self, executor_dependencies
    ):
        # A recoverable InferenceError is also a PauseException, so it must
        # propagate as a graceful interrupt: the step's failure is still
        # published for observation (InferenceStepFailed), but the run is NOT
        # reported as failed (no InferenceFailed), and the original typed error
        # propagates so glyff can leave the step resumable rather than engraving
        # it.
        from sefia.exceptions import InvalidInferenceResponseError

        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        error = InvalidInferenceResponseError("malformed response")
        mock_strategy.decide_next_step.side_effect = error

        executor = InferenceExecutor(
            sample_func,
            ("dummy_arg",),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        with pytest.raises(InvalidInferenceResponseError, match="malformed response"):
            await executor.run()

        published = [call.args[0] for call in mock_publisher.publish.call_args_list]
        step_failures = [
            e for e in published if isinstance(e, events.InferenceStepFailed)
        ]
        assert len(step_failures) == 1
        assert step_failures[0].error is error
        # The run itself is not reported as failed — it is a recoverable yield.
        assert not any(isinstance(e, events.InferenceFailed) for e in published)

    async def test_handler_yield_on_failed_step_is_isolated(
        self, executor_dependencies
    ):
        # An observation handler cannot steer control flow: if it reacts to
        # InferenceStepFailed by raising PauseException, the publisher isolates
        # it, so the original error propagates and is engraved as a genuine
        # failure. Resumable interrupts must come from the control layer (tools),
        # not from observers.
        mock_strategy, mock_collector, _, non_engrave = executor_dependencies
        mock_strategy.decide_next_step.side_effect = ValueError("transient")

        class InterruptOnFailure(EventHandler[events.InferenceStepFailed]):
            async def handle(self, event):
                raise PauseException("interrupted for resume")

        publisher = EventPublisher([InterruptOnFailure()])
        executor = InferenceExecutor(
            sample_func,
            ("dummy_arg",),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            publisher,
        )

        with pytest.raises(ValueError, match="transient"):
            await executor.run()

    async def test_concurrent_tools_overlap_within_a_batch(
        self, executor_dependencies
    ):
        # Two calls to @concurrent tools in one decision run overlapped: the
        # first tool blocks until the second one has run, which can only
        # complete if the executor does not serialize the batch.
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.side_effect = [
            ToolCallDecision(
                calls=[
                    ToolCallRequest(id="1", name="wait_for_peer", arguments={}),
                    ToolCallRequest(id="2", name="release_peer", arguments={}),
                ]
            ),
            ResultDecision(result="done"),
        ]

        peer_ran = asyncio.Event()

        async def wait_for_peer() -> str:
            await asyncio.wait_for(peer_ran.wait(), timeout=5)
            return "waited"

        async def release_peer() -> str:
            peer_ran.set()
            return "released"

        tool_registry = ToolRegistry()
        tool_registry.add(wait_for_peer, name="wait_for_peer", concurrent=True)
        tool_registry.add(release_peer, name="release_peer", concurrent=True)
        mock_collector.collect.return_value = tool_registry

        executor = InferenceExecutor(
            sample_func_with_self,
            (object(), "value"),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        result = await executor.run()

        assert result == "done"
        # Results come back in request order even though the first call
        # finished last.
        history = mock_strategy.decide_next_step.call_args_list[1].kwargs["history"]
        results = [item for item in history if isinstance(item, ToolCallResult)]
        assert [(r.tool_call_id, r.result) for r in results] == [
            ("1", "waited"),
            ("2", "released"),
        ]

    async def test_unmarked_tools_stay_strictly_serial(self, executor_dependencies):
        # Tools without @concurrent keep today's behavior: each call starts
        # only after the previous one completed.
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.side_effect = [
            ToolCallDecision(
                calls=[
                    ToolCallRequest(id="1", name="tool_a", arguments={}),
                    ToolCallRequest(id="2", name="tool_b", arguments={}),
                ]
            ),
            ResultDecision(result="done"),
        ]

        timeline: list[str] = []

        async def tool_a() -> str:
            timeline.append("a:start")
            await asyncio.sleep(0)
            timeline.append("a:end")
            return "a"

        async def tool_b() -> str:
            timeline.append("b:start")
            await asyncio.sleep(0)
            timeline.append("b:end")
            return "b"

        tool_registry = ToolRegistry()
        tool_registry.add(tool_a, name="tool_a")
        tool_registry.add(tool_b, name="tool_b")
        mock_collector.collect.return_value = tool_registry

        executor = InferenceExecutor(
            sample_func_with_self,
            (object(), "value"),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        await executor.run()

        assert timeline == ["a:start", "a:end", "b:start", "b:end"]

    async def test_serial_call_is_a_barrier_between_concurrent_calls(
        self, executor_dependencies
    ):
        # In [concurrent, serial, concurrent], the serial call starts only
        # after the first completed and the last starts only after the serial
        # one completed.
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.side_effect = [
            ToolCallDecision(
                calls=[
                    ToolCallRequest(id="1", name="conc_a", arguments={}),
                    ToolCallRequest(id="2", name="serial_s", arguments={}),
                    ToolCallRequest(id="3", name="conc_b", arguments={}),
                ]
            ),
            ResultDecision(result="done"),
        ]

        timeline: list[str] = []

        def make_tool(label: str):
            async def tool() -> str:
                timeline.append(f"{label}:start")
                await asyncio.sleep(0)
                timeline.append(f"{label}:end")
                return label

            return tool

        tool_registry = ToolRegistry()
        tool_registry.add(make_tool("a"), name="conc_a", concurrent=True)
        tool_registry.add(make_tool("s"), name="serial_s")
        tool_registry.add(make_tool("b"), name="conc_b", concurrent=True)
        mock_collector.collect.return_value = tool_registry

        executor = InferenceExecutor(
            sample_func_with_self,
            (object(), "value"),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        await executor.run()

        assert timeline == [
            "a:start",
            "a:end",
            "s:start",
            "s:end",
            "b:start",
            "b:end",
        ]

    async def test_pause_lets_concurrent_siblings_finish(self, executor_dependencies):
        # A PauseException interrupts the batch, but overlapped siblings run
        # to completion first (an engraved sibling's finished work must be
        # committed before the run pauses).
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.return_value = ToolCallDecision(
            calls=[
                ToolCallRequest(id="1", name="pausing", arguments={}),
                ToolCallRequest(id="2", name="sibling", arguments={}),
            ]
        )

        sibling_finished = False

        async def pausing() -> str:
            raise PauseException("needs input")

        async def sibling() -> str:
            nonlocal sibling_finished
            await asyncio.sleep(0)
            sibling_finished = True
            return "ok"

        tool_registry = ToolRegistry()
        tool_registry.add(pausing, name="pausing", concurrent=True)
        tool_registry.add(sibling, name="sibling", concurrent=True)
        mock_collector.collect.return_value = tool_registry

        executor = InferenceExecutor(
            sample_func_with_self,
            (object(), "value"),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        with pytest.raises(PauseException, match="needs input"):
            await executor.run()

        assert sibling_finished

    async def test_earliest_pause_in_request_order_wins(self, executor_dependencies):
        # When several overlapped calls pause, the one earliest in request
        # order propagates — even if it was raised last in wall-clock time —
        # so the escaping exception is deterministic.
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.return_value = ToolCallDecision(
            calls=[
                ToolCallRequest(id="1", name="pause_late", arguments={}),
                ToolCallRequest(id="2", name="pause_early", arguments={}),
            ]
        )

        second_paused = asyncio.Event()

        async def pause_late() -> str:
            await asyncio.wait_for(second_paused.wait(), timeout=5)
            raise PauseException("first in request order")

        async def pause_early() -> str:
            second_paused.set()
            raise PauseException("second in request order")

        tool_registry = ToolRegistry()
        tool_registry.add(pause_late, name="pause_late", concurrent=True)
        tool_registry.add(pause_early, name="pause_early", concurrent=True)
        mock_collector.collect.return_value = tool_registry

        executor = InferenceExecutor(
            sample_func_with_self,
            (object(), "value"),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        with pytest.raises(PauseException, match="first in request order"):
            await executor.run()

    async def test_tool_failure_in_concurrent_batch_stays_isolated(
        self, executor_dependencies
    ):
        # An ordinary tool failure inside an overlapped batch is stringified
        # into its own slot; siblings are unaffected and the run continues.
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.side_effect = [
            ToolCallDecision(
                calls=[
                    ToolCallRequest(id="1", name="boom", arguments={}),
                    ToolCallRequest(id="2", name="fine", arguments={}),
                ]
            ),
            ResultDecision(result="recovered"),
        ]

        async def boom() -> str:
            raise ValueError("kaboom")

        async def fine() -> str:
            return "ok"

        tool_registry = ToolRegistry()
        tool_registry.add(boom, name="boom", concurrent=True)
        tool_registry.add(fine, name="fine", concurrent=True)
        mock_collector.collect.return_value = tool_registry

        executor = InferenceExecutor(
            sample_func_with_self,
            (object(), "value"),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        result = await executor.run()

        assert result == "recovered"
        history = mock_strategy.decide_next_step.call_args_list[1].kwargs["history"]
        results = [item for item in history if isinstance(item, ToolCallResult)]
        assert "Error executing tool 'boom'" in results[0].result
        assert results[1].result == "ok"

    async def test_identical_concurrent_calls_run_serially(
        self, executor_dependencies
    ):
        # Two calls with the same tool and arguments never overlap (glyff's
        # sequencer numbers repeated executions of one content key by arrival,
        # so racing duplicates would make replay assignment nondeterministic).
        # A third call with different arguments still overlaps with them.
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.side_effect = [
            ToolCallDecision(
                calls=[
                    ToolCallRequest(id="1", name="fetch", arguments={"key": "same"}),
                    ToolCallRequest(id="2", name="fetch", arguments={"key": "same"}),
                    ToolCallRequest(id="3", name="fetch", arguments={"key": "other"}),
                ]
            ),
            ResultDecision(result="done"),
        ]

        active_same = 0
        max_active_same = 0
        saw_other_during_same = False
        active_other = 0

        async def fetch(key: str) -> str:
            nonlocal active_same, max_active_same, saw_other_during_same, active_other
            if key == "same":
                active_same += 1
                max_active_same = max(max_active_same, active_same)
            else:
                active_other += 1
            await asyncio.sleep(0.01)
            if key == "same" and active_other:
                saw_other_during_same = True
            if key == "same":
                active_same -= 1
            else:
                active_other -= 1
            return key

        tool_registry = ToolRegistry()
        tool_registry.add(fetch, name="fetch", concurrent=True)
        mock_collector.collect.return_value = tool_registry

        executor = InferenceExecutor(
            sample_func_with_self,
            (object(), "value"),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        await executor.run()

        assert max_active_same == 1
        assert saw_other_during_same

    async def test_retry_middleware_publishes_attempt_start_per_attempt(
        self, executor_dependencies
    ):
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.side_effect = [
            ValueError("flaky inference"),
            ResultDecision(result="attempt 2 succeeds"),
        ]

        executor = InferenceExecutor(
            sample_func,
            ("dummy_arg",),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
            inference_middlewares=[_Retrier(max_retries=1)],
        )
        result = await executor.run()

        assert result == "attempt 2 succeeds"
        attempt_events = [
            call.args[0]
            for call in mock_publisher.publish.call_args_list
            if isinstance(call.args[0], AttemptStart)
        ]
        assert len(attempt_events) == 2
