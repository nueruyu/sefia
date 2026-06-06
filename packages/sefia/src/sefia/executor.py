import inspect
from typing import Any, Callable, ParamSpec, TypeVar

from glyff.exceptions import YieldException

from . import events
from .event_publisher import EventPublisher
from .handlers.retry import RequestInferenceRetry
from .interfaces import InferenceStrategy, ToolCollector
from .models import (
    FinalAnswerDecision,
    HistoryItem,
    InferenceDecision,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
    ToolRegistry,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _wrap(f: Callable[_P, _R], decorator) -> Callable[_P, _R]:
    return decorator(f)


class InferenceExecutor:
    """
    Orchestrates the inference loop for a single @infer call.
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
    ):
        self.args = args
        self.kwargs = kwargs
        self.strategy = inference_strategy
        self.tool_collector = tool_collector
        self.publisher = publisher
        self._tool_registry: ToolRegistry | None = None

        type_hints = inspect.get_annotations(func, eval_str=True)
        self.return_type = type_hints.get("return", Any)
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
                history=history,
                tools=tools,
                output_type=self.return_type,
                publisher=self.publisher,
            )
        except Exception as e:
            # This method is engraved, so any exception that escapes it is
            # persisted by glyff as a permanent FAILED record. sefia does not
            # decide whether the failure is recoverable: it publishes the error
            # and lets a handler decide. A handler may raise YieldException to
            # interrupt gracefully and keep the step resumable (nothing is
            # engraved); if none does, the original exception is re-raised and
            # engraved as a genuine failure.
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
                    result = await tool_func(**call.arguments)
                    output = result
                    await self.publisher.publish(
                        events.AfterToolCall(tool_call=call, result=result)
                    )
                except YieldException:
                    raise
                except Exception as e:
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
        Runs the inference process, handling retries as requested by handlers.
        """
        await self.publisher.publish(
            events.InferenceStart(
                func_name=self.func_name,
                args=self.args,
                kwargs=self.kwargs,
            )
        )

        while True:
            try:
                await self.publisher.publish(events.AttemptStart())
                result = await self._attempt_inference()
                await self.publisher.publish(events.InferenceEnd(result=result))
                return result
            except RequestInferenceRetry:
                continue
            except YieldException:
                raise
            except Exception as e:
                try:
                    await self.publisher.publish(events.InferenceFailed(error=e))
                except RequestInferenceRetry:
                    continue
                raise

    async def _attempt_inference(self) -> Any:
        """Executes a single attempt of the inference loop."""

        instance = self.arguments.get("self")
        self._tool_registry = (
            self.tool_collector.collect(instance) if instance is not None else None
        )
        tool_schemas = (
            self._tool_registry.get_all_schemas() if self._tool_registry else []
        )

        history: list[HistoryItem] = []
        max_turns = 25

        for _ in range(max_turns):
            decision = await self._next_step_engraved(history, tool_schemas)

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

        raise RuntimeError(f"Inference did not complete within {max_turns} turns.")
