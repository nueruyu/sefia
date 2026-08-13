from typing import Any, Awaitable, Callable

from . import events
from ._history import StepHistory
from ._interfaces import InferenceStrategy
from ._interfaces.history_storage import HistorySnapshot, HistoryStorage
from ._interfaces.middleware import (
    InferenceContext,
    InferenceMiddleware,
    StepContext,
    StepMiddleware,
)
from ._tool_execution import call_tools
from ._tool_system import ToolCollector, ToolRegistry
from .event_system import EventPublisher
from .exceptions import PauseException
from .inference import (
    ResultDecision,
    FunctionInfo,
    InferenceDecision,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)


def _compose(
    middlewares: list, ctx: Any, core: Callable[[], Awaitable[Any]]
) -> Callable[[], Awaitable[Any]]:
    """
    Compose ``middlewares`` into an onion around ``core``.

    Middlewares are applied so the first in the list is the outermost layer.
    A middleware receives ``nxt`` as the next layer and may call it once, call it
    again for retry behavior, or short-circuit by raising an exception.
    """
    nxt = core
    for middleware in reversed(middlewares):
        nxt = _layer(middleware, ctx, nxt)
    return nxt


def _layer(middleware, ctx, nxt: Callable[[], Awaitable[Any]]):
    async def call() -> Any:
        return await middleware.wrap(ctx, nxt)

    return call


class InferenceExecutor:
    """
    Orchestrates the inference loop for a single @infer call.

    The executor owns the inference lifecycle and the inner step loop, and wraps
    the run with configured middleware. Middleware keeps control flow explicit
    and separate from observation (which flows through the event publisher).
    """

    def __init__(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        inference_strategy: InferenceStrategy,
        tool_collector: ToolCollector,
        engrave: Callable[[str, Callable[..., Any]], Callable[..., Any]],
        publisher: EventPublisher,
        history_storage: HistoryStorage,
        inference_middlewares: list[InferenceMiddleware] | None = None,
        step_middlewares: list[StepMiddleware] | None = None,
    ):
        self.func_info = FunctionInfo.create(func, args, kwargs)
        self.strategy = inference_strategy
        self.publisher = publisher
        self._storage = history_storage
        self._history = StepHistory()
        self._completed_steps = 0
        self._inference_middlewares = inference_middlewares or []
        self._step_middlewares = step_middlewares or []

        self._tool_registry: ToolRegistry = tool_collector.collect(
            self.func_info.capabilities
        )

        self._next_step_engraved = engrave("inference_step", self._next_step)
        self._call_tools_engraved = engrave("tool_batch", self._call_tools)

    async def _next_step(self, step: int) -> InferenceDecision:
        """One engraved inference-strategy call, keyed on the step index (not
        the history) so the durable key stays O(1) and survives compaction."""
        history = self._history.items
        await self.publisher.publish(
            events.BeforeInferenceStep(
                history=history,
                tool_names=self._tool_registry.get_names(),
            )
        )

        try:
            decision = await self.strategy.decide_next_step(
                function_info=self.func_info,
                history=history,
                tools=self._tool_registry,
                publisher=self.publisher,
            )
        except Exception as e:
            # Observation only, then re-raise: glyff leaves the step resumable,
            # and run() classifies it as a pause or a failure upstream.
            await self.publisher.publish(events.InferenceStepFailed(error=e))
            raise

        await self.publisher.publish(events.AfterInferenceStep(decision=decision))
        return decision

    async def _call_tools(
        self, tool_calls: list[ToolCallRequest]
    ) -> list[ToolCallResult]:
        """Internal engraved method for executing a batch of tool calls."""
        return await call_tools(tool_calls, self._tool_registry, self.publisher)

    async def run(self) -> Any:
        """
        Runs the inference process.

        Inference middleware wraps the attempt factory. A retry middleware may
        call the wrapped function more than once; any exception or genuine
        failure that escapes middleware propagates out.
        """
        await self.publisher.publish(
            events.InferenceStart(
                func_name=self.func_info.qualname,
                args=self.func_info.args,
                kwargs=self.func_info.kwargs,
            )
        )

        ctx = InferenceContext(
            func_name=self.func_info.qualname,
            args=self.func_info.args,
            kwargs=self.func_info.kwargs,
        )

        async def core() -> Any:
            await self.publisher.publish(events.AttemptStart())
            return await self._attempt_inference()

        chain = _compose(self._inference_middlewares, ctx, core)

        try:
            result = await chain()
            await self.publisher.publish(events.InferenceEnd(result=result))
            return result
        except PauseException:
            raise
        except Exception as e:
            await self.publisher.publish(events.InferenceFailed(error=e))
            raise

    async def _attempt_inference(self) -> Any:
        """Executes a single attempt of the inference loop, owning the step loop."""
        snapshot = await self._storage.load()
        self._history = StepHistory(snapshot.items)
        self._completed_steps = snapshot.completed_steps

        while True:
            step = self._completed_steps
            await self.publisher.publish(
                events.StepStarted(step=step, history=self._history.items)
            )

            step_ctx = StepContext(
                step=step,
                history=self._history,
                tool_registry=self._tool_registry,
            )

            async def core() -> InferenceDecision:
                # Persist a compaction before the model call, so a resume loads
                # it instead of re-running the compactor.
                await self._save_history()
                return await self._next_step_engraved(step)

            step_chain = _compose(self._step_middlewares, step_ctx, core)
            decision = await step_chain()
            await self._save_history()

            if isinstance(decision, ResultDecision):
                return decision.result

            if isinstance(decision, ToolCallDecision):
                if decision.calls:
                    tool_results = await self._call_tools_engraved(decision.calls)
                else:
                    tool_results = []
                # Persist only after the engraved calls commit, so a crash
                # resumes from the previous snapshot and replays the step.
                self._history.extend([decision, *tool_results])
                self._completed_steps += 1
                await self._save_history()
            else:
                raise TypeError(f"Unknown decision type: {type(decision)}")

    async def _save_history(self) -> None:
        if not self._history.dirty:
            return
        await self._storage.save(
            HistorySnapshot(self._history.items, self._completed_steps)
        )
        self._history.mark_persisted()
