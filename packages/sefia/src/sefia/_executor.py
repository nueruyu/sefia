import inspect
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

from glyff.exceptions import YieldException

from . import events
from ._interfaces import InferenceStrategy
from ._interfaces.middleware import (
    InferenceContext,
    InferenceMiddleware,
    StepContext,
    StepMiddleware,
)
from ._tool_system import ToolCollector, ToolRegistry
from .event_system import EventPublisher
from .inference import (
    FinalAnswerDecision,
    FunctionInfo,
    HistoryItem,
    InferenceDecision,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _wrap(f: Callable[_P, _R], decorator) -> Callable[_P, _R]:
    return decorator(f)


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
        engrave: Callable[[Any], Any],
        publisher: EventPublisher,
        inference_middlewares: list[InferenceMiddleware] | None = None,
        step_middlewares: list[StepMiddleware] | None = None,
    ):
        self.func_info = FunctionInfo.create(func, args, kwargs)
        self.strategy = inference_strategy
        self.publisher = publisher
        self._inference_middlewares = inference_middlewares or []
        self._step_middlewares = step_middlewares or []

        instance = self.func_info.instance
        if instance is not None:
            self._tool_registry: ToolRegistry = tool_collector.collect(instance)
        else:
            self._tool_registry = ToolRegistry()
        self._tool_schemas: list[dict] = [
            t.schema for t in self._tool_registry.get_all()
        ]

        self._next_step_engraved = _wrap(self._next_step, engrave)
        self._call_tools_engraved = _wrap(self._call_tools, engrave)

    async def _next_step(self, history: list[HistoryItem]) -> InferenceDecision:
        """Internal engraved method for a single inference strategy call."""
        await self.publisher.publish(
            events.BeforeInferenceStep(history=history, tools=self._tool_schemas)
        )

        try:
            decision = await self.strategy.decide_next_step(
                function_info=self.func_info,
                history=history,
                tools=self._tool_schemas,
                publisher=self.publisher,
            )
        except Exception as e:
            # This method is engraved. The failure is published for observation
            # (handlers cannot change the outcome — the publisher isolates their
            # exceptions), then the original exception is re-raised. What glyff
            # does next depends on the exception type: a recoverable
            # InferenceError is also a YieldException, so glyff leaves the step
            # resumable instead of engraving it — a transient hiccup or invalid
            # LLM response never poisons the step, and a re-invocation re-runs it.
            # Any other exception is engraved as a genuine, permanent FAILED
            # record. (Resumable interrupts otherwise come from the
            # control/execution layer, e.g. a tool raising YieldException, not
            # from observation handlers.)
            await self.publisher.publish(events.InferenceStepFailed(error=e))
            raise

        await self.publisher.publish(events.AfterInferenceStep(decision=decision))
        return decision

    async def _call_tools(
        self, tool_calls: list[ToolCallRequest]
    ) -> list[ToolCallResult]:
        """Internal engraved method for executing a batch of tool calls."""
        tool_results: list[ToolCallResult] = []
        for call in tool_calls:
            await self.publisher.publish(events.BeforeToolCall(tool_call=call))
            tool_name = call.name
            tool_info = self._tool_registry.get(tool_name)

            if not tool_info:
                output = f"Error: Tool '{tool_name}' not found."
                await self.publisher.publish(
                    events.ToolExecutionFailed(
                        tool_call=call,
                        error=RuntimeError(f"Tool '{tool_name}' not found."),
                    )
                )
            else:
                try:
                    tool_func = tool_info.function
                    result = tool_func(**call.arguments)
                    if inspect.isawaitable(result):
                        result = await result
                    output = result
                    await self.publisher.publish(
                        events.AfterToolCall(tool_call=call, result=result)
                    )
                except YieldException:
                    raise
                except Exception as e:
                    # A tool failure is never a retryable inference failure: we
                    # stringify it into the history and feed it back to the model
                    # so it can recover, then keep going.
                    await self.publisher.publish(
                        events.ToolExecutionFailed(tool_call=call, error=e)
                    )
                    output = (
                        f"Error executing tool '{tool_name}': {type(e).__name__}({e})"
                    )
            tool_results.append(ToolCallResult(tool_call_id=call.id, result=output))
        return tool_results

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
        except YieldException:
            raise
        except Exception as e:
            await self.publisher.publish(events.InferenceFailed(error=e))
            raise

    async def _attempt_inference(self) -> Any:
        """Executes a single attempt of the inference loop, owning the step loop."""
        history: list[HistoryItem] = []
        step = 0

        while True:
            await self.publisher.publish(events.StepStarted(step=step, history=history))

            step_ctx = StepContext(step=step, history=history)

            async def core() -> InferenceDecision:
                return await self._next_step_engraved(history)

            step_chain = _compose(self._step_middlewares, step_ctx, core)
            decision = await step_chain()
            step += 1

            if isinstance(decision, FinalAnswerDecision):
                return decision.answer

            if isinstance(decision, ToolCallDecision):
                history.append(decision)
                if not decision.calls:
                    continue

                tool_results = await self._call_tools_engraved(decision.calls)
                history.extend(tool_results)
            else:
                raise TypeError(f"Unknown decision type: {type(decision)}")
