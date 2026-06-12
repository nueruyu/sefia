import inspect
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

from glyff.exceptions import YieldException

from . import events
from ._interfaces import InferenceStrategy
from ._interfaces.middleware import (
    InferenceMiddleware,
    InferenceContext,
    StepContext,
    StepMiddleware,
)
from .event_system import EventPublisher
from .exceptions import RequestInferenceRetry
from .inference import (
    FinalAnswerDecision,
    HistoryItem,
    InferenceDecision,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)
from .tools import ToolCollector, ToolRegistry

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _wrap(f: Callable[_P, _R], decorator) -> Callable[_P, _R]:
    return decorator(f)


def _compose(
    middlewares: list, ctx: Any, core: Callable[[], Awaitable[Any]]
) -> Callable[[], Awaitable[Any]]:
    """
    Compose ``middlewares`` into an onion around ``core``.

    The executor owns the loop; each middleware merely wraps the single unit of
    work (``core``) and delegates to the next layer via ``nxt``. Middlewares are
    applied so the first in the list is the outermost layer.
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

    The executor owns both loops (the outer retry loop and the inner step loop)
    and wraps each unit of work with the configured middleware. Middleware steers
    those loops by raising typed control signals, keeping control flow explicit
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
        self.args = args
        self.kwargs = kwargs
        self.strategy = inference_strategy
        self.tool_collector = tool_collector
        self.publisher = publisher
        self._inference_middlewares = inference_middlewares or []
        self._step_middlewares = step_middlewares or []
        self._tool_registry: ToolRegistry | None = None

        self.type_hints = inspect.get_annotations(func, eval_str=True)
        self.return_type = self.type_hints.get("return", Any)
        self.instructions = inspect.getdoc(func) or "Execute the requested task."
        self.func_name = func.__qualname__

        sig = inspect.signature(func)
        bound_args = sig.bind(*self.args, **self.kwargs)
        bound_args.apply_defaults()
        self.arguments = bound_args.arguments

        self._next_step_engraved = _wrap(self._next_step, engrave)
        self._call_tools_engraved = _wrap(self._call_tools, engrave)

    async def _next_step(
        self, history: list[HistoryItem], tools: list[dict]
    ) -> InferenceDecision:
        """Internal engraved method for a single inference strategy call."""
        await self.publisher.publish(
            events.BeforeInferenceStep(history=history, tools=tools)
        )

        try:
            decision = await self.strategy.decide_next_step(
                instructions=self.instructions,
                arguments=self.arguments,
                argument_type_hints=self.type_hints,
                history=history,
                tools=tools,
                output_type=self.return_type,
                publisher=self.publisher,
            )
        except Exception as e:
            # This method is engraved, so any exception that escapes it is
            # persisted by glyff as a permanent FAILED record. The failure is
            # published for observation (handlers cannot change the outcome — the
            # publisher isolates their exceptions), then the original exception is
            # re-raised and engraved as a genuine failure. Resumable interrupts
            # come from the control/execution layer (e.g. a tool raising
            # YieldException), not from observation handlers.
            await self.publisher.publish(events.InferenceStepFailed(error=e))
            raise

        await self.publisher.publish(events.AfterInferenceStep(decision=decision))
        return decision

    async def _call_tools(
        self, tool_calls: list[ToolCallRequest]
    ) -> list[ToolCallResult]:
        """Internal engraved method for executing a batch of tool calls."""
        assert self._tool_registry is not None
        tool_results: list[ToolCallResult] = []
        for call in tool_calls:
            await self.publisher.publish(events.BeforeToolCall(tool_call=call))
            tool_name = call.name
            tool_info = self._tool_registry.get(tool_name)

            if not tool_info:
                output = f"Error: Tool '{tool_name}' not found."
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
        Runs the inference process, owning the retry loop.

        Inference middleware wraps each attempt. A middleware may raise
        ``RequestInferenceRetry`` to ask for another attempt; any other control
        signal (or genuine failure) propagates out.
        """
        await self.publisher.publish(
            events.InferenceStart(
                func_name=self.func_name,
                args=self.args,
                kwargs=self.kwargs,
            )
        )

        ctx = InferenceContext(
            func_name=self.func_name, args=self.args, kwargs=self.kwargs
        )

        async def core() -> Any:
            return await self._attempt_inference()

        # ctx and core are loop-invariant, so the chain is built once; each retry
        # simply re-invokes it. The middleware instances (and their state, e.g.
        # the retry counter) persist across attempts.
        chain = _compose(self._inference_middlewares, ctx, core)

        while True:
            try:
                await self.publisher.publish(events.AttemptStart())
                result = await chain()
                await self.publisher.publish(events.InferenceEnd(result=result))
                return result
            except RequestInferenceRetry:
                continue
            except YieldException:
                raise
            except Exception as e:
                await self.publisher.publish(events.InferenceFailed(error=e))
                raise

    async def _attempt_inference(self) -> Any:
        """Executes a single attempt of the inference loop, owning the step loop."""

        instance = self.arguments.get("self")
        self._tool_registry = (
            self.tool_collector.collect(instance) if instance is not None else None
        )
        tool_schemas = (
            self._tool_registry.get_all_schemas() if self._tool_registry else []
        )

        history: list[HistoryItem] = []
        step = 0

        while True:
            await self.publisher.publish(events.StepStarted(step=step, history=history))

            step_ctx = StepContext(step=step, history=history)

            async def core() -> InferenceDecision:
                return await self._next_step_engraved(history, tool_schemas)

            step_chain = _compose(self._step_middlewares, step_ctx, core)
            decision = await step_chain()
            step += 1

            if isinstance(decision, FinalAnswerDecision):
                return decision.answer

            if isinstance(decision, ToolCallDecision):
                history.append(decision)
                if not decision.calls:
                    continue

                if self._tool_registry:
                    tool_results = await self._call_tools_engraved(decision.calls)
                    history.extend(tool_results)
                else:
                    first_call = decision.calls[0]
                    await self.publisher.publish(
                        events.ToolExecutionFailed(
                            tool_call=first_call,
                            error=RuntimeError("No tools available"),
                        )
                    )
                    history.append(
                        ToolCallResult(
                            tool_call_id=first_call.id,
                            result="Error: No tools are available to execute.",
                        )
                    )
            else:
                raise TypeError(f"Unknown decision type: {type(decision)}")
