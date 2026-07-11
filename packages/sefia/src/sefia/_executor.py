import asyncio
import json
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

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
from .exceptions import PauseException
from .inference import (
    ResultDecision,
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

        self._next_step_engraved = _wrap(self._next_step, engrave)
        self._call_tools_engraved = _wrap(self._call_tools, engrave)

    async def _next_step(self, history: list[HistoryItem]) -> InferenceDecision:
        """Internal engraved method for a single inference strategy call."""
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
            # This method is engraved. The failure is published for observation
            # (handlers cannot change the outcome — the publisher isolates their
            # exceptions), then the original exception is re-raised. glyff leaves
            # any interrupted execution in its STARTED state, so a re-invocation
            # re-runs it regardless of the exception type. A recoverable
            # InferenceError is also a PauseException, so the executor treats it
            # as a pause (no InferenceFailed) — a transient hiccup or invalid LLM
            # response pauses the run and a re-invocation re-runs the step. Any
            # other exception is reported through InferenceFailed. (Resumable
            # interrupts otherwise come from the control/execution layer, e.g. a
            # tool raising PauseException, not from observation handlers.)
            await self.publisher.publish(events.InferenceStepFailed(error=e))
            raise

        await self.publisher.publish(events.AfterInferenceStep(decision=decision))
        return decision

    async def _call_tools(
        self, tool_calls: list[ToolCallRequest]
    ) -> list[ToolCallResult]:
        """Internal engraved method for executing a batch of tool calls.

        The batch is walked in request order. Consecutive calls to tools
        declared ``@concurrent`` run overlapped; any other call runs strictly
        serially (it starts only after everything before it completed, and
        everything after it waits). Results always come back in request order,
        so history — and therefore glyff replay — is independent of completion
        order.
        """
        results: dict[int, ToolCallResult] = {}
        index = 0
        while index < len(tool_calls):
            if not self._allows_concurrency(tool_calls[index]):
                results[index] = await self._call_one(tool_calls[index])
                index += 1
                continue
            end = index + 1
            while end < len(tool_calls) and self._allows_concurrency(tool_calls[end]):
                end += 1
            results.update(
                await self._call_concurrent_group(
                    list(enumerate(tool_calls[index:end], start=index))
                )
            )
            index = end
        return [results[i] for i in range(len(tool_calls))]

    def _allows_concurrency(self, call: ToolCallRequest) -> bool:
        tool = self._tool_registry.get(call.name)
        return tool is not None and tool.concurrent

    async def _call_concurrent_group(
        self, indexed_calls: list[tuple[int, ToolCallRequest]]
    ) -> dict[int, ToolCallResult]:
        """Run one batch segment of concurrency-safe calls, overlapped.

        Identical calls (same tool and arguments) share a lane and run in
        request order: glyff's sequencer numbers repeated executions of the
        same content key by arrival, so letting duplicates race would let a
        live run and its replay assign results to occurrences differently.

        A ``PauseException`` (or any unexpected error) stops only its own
        lane; the other lanes still run to completion — an engraved sibling's
        finished work is committed and survives the pause — and then the
        failure of the earliest call in request order propagates, so which
        exception escapes does not depend on completion order.
        """
        if len(indexed_calls) == 1:
            index, call = indexed_calls[0]
            return {index: await self._call_one(call)}

        lanes: dict[str, list[tuple[int, ToolCallRequest]]] = {}
        for index, call in indexed_calls:
            args_key = json.dumps(call.arguments, sort_keys=True, default=repr)
            lanes.setdefault(f"{call.name}:{args_key}", []).append((index, call))

        results: dict[int, ToolCallResult] = {}
        failures: list[tuple[int, Exception]] = []

        async def run_lane(lane: list[tuple[int, ToolCallRequest]]) -> None:
            for index, call in lane:
                try:
                    results[index] = await self._call_one(call)
                except Exception as e:
                    failures.append((index, e))
                    return

        await asyncio.gather(*(run_lane(lane) for lane in lanes.values()))

        if failures:
            _, error = min(failures, key=lambda failure: failure[0])
            raise error
        return results

    async def _call_one(self, call: ToolCallRequest) -> ToolCallResult:
        """Execute a single tool call and fold any tool failure into its result."""
        await self.publisher.publish(events.BeforeToolCall(tool_call=call))
        tool_name = call.name
        tool = self._tool_registry.get(tool_name)

        if not tool:
            result = f"Error: Tool '{tool_name}' not found."
            await self.publisher.publish(
                events.ToolExecutionFailed(
                    tool_call=call,
                    error=RuntimeError(f"Tool '{tool_name}' not found."),
                )
            )
        else:
            try:
                result = await tool.invoke(call.arguments)
                await self.publisher.publish(
                    events.AfterToolCall(tool_call=call, result=result)
                )
            except PauseException:
                raise
            except Exception as e:
                # A tool failure is never a retryable inference failure: we
                # stringify it into the history and feed it back to the model
                # so it can recover, then keep going.
                await self.publisher.publish(
                    events.ToolExecutionFailed(tool_call=call, error=e)
                )
                result = f"Error executing tool '{tool_name}': {type(e).__name__}({e})"
        return ToolCallResult(tool_call_id=call.id, result=result)

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
        history: list[HistoryItem] = []
        step = 0

        while True:
            await self.publisher.publish(events.StepStarted(step=step, history=history))

            step_ctx = StepContext(
                step=step,
                history=history,
                tool_registry=self._tool_registry,
            )

            async def core() -> InferenceDecision:
                return await self._next_step_engraved(history)

            step_chain = _compose(self._step_middlewares, step_ctx, core)
            decision = await step_chain()
            step += 1

            if isinstance(decision, ResultDecision):
                return decision.result

            if isinstance(decision, ToolCallDecision):
                history.append(decision)
                if not decision.calls:
                    continue

                tool_results = await self._call_tools_engraved(decision.calls)
                history.extend(tool_results)
            else:
                raise TypeError(f"Unknown decision type: {type(decision)}")
