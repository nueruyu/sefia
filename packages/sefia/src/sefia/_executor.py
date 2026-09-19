from collections.abc import Sequence
from typing import Any, Awaitable, Callable, cast, overload

from . import events
from ._history import StepHistory
from ._interfaces import InferenceStrategy
from ._interfaces.history_storage import HistorySnapshot, HistoryStorage
from ._interfaces.middleware import (
    DecisionContext,
    DecisionMiddleware,
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
    StepDecision,
    ToolCallsDecision,
    ToolCallRequest,
    ToolCallResult,
)


@overload
def _compose(
    middlewares: Sequence[InferenceMiddleware],
    ctx: InferenceContext,
    core: Callable[[], Awaitable[Any]],
) -> Callable[[], Awaitable[Any]]: ...


@overload
def _compose(
    middlewares: Sequence[StepMiddleware],
    ctx: StepContext,
    core: Callable[[], Awaitable[StepDecision]],
) -> Callable[[], Awaitable[StepDecision]]: ...


@overload
def _compose(
    middlewares: Sequence[DecisionMiddleware],
    ctx: DecisionContext,
    core: Callable[[], Awaitable[StepDecision]],
) -> Callable[[], Awaitable[StepDecision]]: ...


def _compose(
    middlewares: Sequence[InferenceMiddleware | StepMiddleware | DecisionMiddleware],
    ctx: InferenceContext | StepContext | DecisionContext,
    core: Callable[[], Awaitable[Any]],
) -> Callable[[], Awaitable[Any]]:
    """
    Compose ``middlewares`` into an onion around ``core``.

    Middlewares are applied so the first in the list is the outermost layer.
    A middleware receives ``nxt`` as the next layer and may call it once, call it
    again for retry behavior, or short-circuit by returning or raising.
    """
    nxt = core
    for middleware in reversed(middlewares):
        nxt = _layer(middleware, ctx, nxt)
    return nxt


def _layer(
    middleware: InferenceMiddleware | StepMiddleware | DecisionMiddleware,
    ctx: InferenceContext | StepContext | DecisionContext,
    nxt: Callable[[], Awaitable[Any]],
) -> Callable[[], Awaitable[Any]]:
    async def call() -> Any:
        if isinstance(middleware, InferenceMiddleware) and isinstance(
            ctx, InferenceContext
        ):
            return await middleware.wrap(ctx, nxt)
        if isinstance(middleware, StepMiddleware) and isinstance(ctx, StepContext):
            return await middleware.wrap(ctx, nxt)
        if isinstance(middleware, DecisionMiddleware) and isinstance(
            ctx, DecisionContext
        ):
            return await middleware.wrap(ctx, nxt)
        raise TypeError("Middleware and context types do not match.")

    return call


def _require_step_decision(decision: object) -> StepDecision:
    if isinstance(decision, (ResultDecision, ToolCallsDecision)):
        return decision
    raise TypeError(f"Unknown decision type: {type(decision)}")


def _require_tool_calls_decision(decision: object) -> ToolCallsDecision:
    if isinstance(decision, ToolCallsDecision):
        return decision
    raise TypeError(f"Unknown decision type: {type(decision)}")


class InferenceExecutor:
    """
    Orchestrates the inference loop for a single @infer call.

    The executor owns the inference lifecycle and the inner step loop, and wraps
    the run with configured middleware. Middleware keeps control flow explicit
    and separate from observation (which flows through the event publisher).
    """

    def __init__(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        inference_strategy: InferenceStrategy,
        tool_collector: ToolCollector,
        engrave: Callable[[str, Callable[..., Any]], Callable[..., Any]],
        publisher: EventPublisher,
        history_storage: HistoryStorage,
        inference_middlewares: list[InferenceMiddleware] | None = None,
        step_middlewares: list[StepMiddleware] | None = None,
        decision_middlewares: list[DecisionMiddleware] | None = None,
    ):
        self.func_info = FunctionInfo.create(func, args, kwargs)
        self.strategy = inference_strategy
        self.publisher = publisher
        self._storage = history_storage
        self._history = StepHistory()
        self._completed_steps = 0
        self._inference_middlewares = inference_middlewares or []
        self._step_middlewares = step_middlewares or []
        self._decision_middlewares = decision_middlewares or []

        self._tool_registry: ToolRegistry = tool_collector.collect(
            self.func_info.capabilities
        )

        self._next_step_engraved = cast(
            Callable[[int], Awaitable[StepDecision]],
            engrave("inference.step", self._next_step),
        )
        self._call_tools_engraved = cast(
            Callable[[list[ToolCallRequest]], Awaitable[list[ToolCallResult]]],
            engrave("inference.tool_calls", self._call_tools),
        )

    async def _next_step(self, step: int) -> StepDecision:
        """One engraved decision execution, keyed on the step index (not
        the history) so the durable key stays O(1) and survives compaction."""
        history = self._history.items
        await self.publisher.publish(
            events.BeforeInferenceStep(
                history=history,
                tool_names=self._tool_registry.get_names(),
            )
        )

        ctx = DecisionContext(
            step=step,
            function_info=self.func_info,
            history=history,
        )

        async def core() -> StepDecision:
            return await self.strategy.decide_next_step(
                function_info=self.func_info,
                history=history,
                tools=self._tool_registry,
                publisher=self.publisher,
            )

        chain = _compose(self._decision_middlewares, ctx, core)

        try:
            decision = _require_step_decision(await chain())
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

            async def core() -> StepDecision:
                # Persist a compaction before the model call, so a resume loads
                # it instead of re-running the compactor.
                await self._save_history()
                return await self._next_step_engraved(step)

            step_chain = _compose(self._step_middlewares, step_ctx, core)
            decision = await step_chain()
            await self._save_history()

            if isinstance(decision, ResultDecision):
                return decision.result

            decision = _require_tool_calls_decision(decision)
            if decision.calls:
                tool_results = await self._call_tools_engraved(decision.calls)
            else:
                tool_results = []
            # Persist only after the engraved calls commit, so a crash
            # resumes from the previous snapshot and replays the step.
            self._history.extend([decision, *tool_results])
            self._completed_steps += 1
            await self._save_history()

    async def _save_history(self) -> None:
        if not self._history.dirty:
            return
        await self._storage.save(
            HistorySnapshot(self._history.items, self._completed_steps)
        )
        self._history.mark_persisted()
