import asyncio
import json

from . import events
from ._tool_context import serving_tool_call
from ._tool_system import ToolRegistry
from .event_system import EventPublisher
from .exceptions import PauseException
from .inference import ToolCallRequest, ToolCallResult


async def call_tools(
    tool_calls: list[ToolCallRequest],
    registry: ToolRegistry,
    publisher: EventPublisher,
) -> list[ToolCallResult]:
    """Execute a decision's tool calls, returning results in request order.

    The batch runs serially except that consecutive calls to ``@concurrent``
    tools overlap; an unmarked call is a barrier. Results are ordered by
    request, not completion, so history stays stable for replay.
    """
    results: dict[int, ToolCallResult] = {}
    index = 0
    while index < len(tool_calls):
        if not _allows_concurrency(tool_calls[index], registry):
            results[index] = await _call_one(tool_calls[index], registry, publisher)
            index += 1
            continue
        end = index + 1
        while end < len(tool_calls) and _allows_concurrency(tool_calls[end], registry):
            end += 1
        results.update(
            await _call_group(
                list(enumerate(tool_calls[index:end], start=index)),
                registry,
                publisher,
            )
        )
        index = end
    return [results[i] for i in range(len(tool_calls))]


def _allows_concurrency(call: ToolCallRequest, registry: ToolRegistry) -> bool:
    tool = registry.get(call.name)
    return tool is not None and tool.concurrent


async def _call_group(
    indexed_calls: list[tuple[int, ToolCallRequest]],
    registry: ToolRegistry,
    publisher: EventPublisher,
) -> dict[int, ToolCallResult]:
    """Run one overlapped batch segment.

    Identical calls (same tool and arguments) share a lane, since glyff
    sequences repeated executions of one content key by arrival order. A
    ``PauseException`` stops only its own lane; the rest run to completion,
    then the failure of the earliest call in request order propagates.
    """
    if len(indexed_calls) == 1:
        index, call = indexed_calls[0]
        return {index: await _call_one(call, registry, publisher)}

    lanes: dict[str, list[tuple[int, ToolCallRequest]]] = {}
    for index, call in indexed_calls:
        args_key = json.dumps(call.arguments, sort_keys=True, default=repr)
        lanes.setdefault(f"{call.name}:{args_key}", []).append((index, call))

    results: dict[int, ToolCallResult] = {}
    failures: list[tuple[int, Exception]] = []

    async def run_lane(lane: list[tuple[int, ToolCallRequest]]) -> None:
        for index, call in lane:
            try:
                results[index] = await _call_one(call, registry, publisher)
            except Exception as e:
                failures.append((index, e))
                return

    await asyncio.gather(*(run_lane(lane) for lane in lanes.values()))

    if failures:
        _, error = min(failures, key=lambda failure: failure[0])
        raise error
    return results


async def _call_one(
    call: ToolCallRequest, registry: ToolRegistry, publisher: EventPublisher
) -> ToolCallResult:
    """Execute a single tool call, folding tool failures into the result."""
    await publisher.publish(events.BeforeToolCall(tool_call=call))
    tool_name = call.name
    tool = registry.get(tool_name)

    if not tool:
        result = f"Error: Tool '{tool_name}' not found."
        await publisher.publish(
            events.ToolExecutionFailed(
                tool_call=call,
                error=RuntimeError(f"Tool '{tool_name}' not found."),
            )
        )
    else:
        try:
            with serving_tool_call(call.id):
                result = await tool.invoke(call.arguments)
            await publisher.publish(events.AfterToolCall(tool_call=call, result=result))
        except PauseException:
            raise
        except Exception as e:
            # A tool failure is never a retryable inference failure: we
            # stringify it into the history and feed it back to the model
            # so it can recover, then keep going.
            await publisher.publish(events.ToolExecutionFailed(tool_call=call, error=e))
            result = f"Error executing tool '{tool_name}': {type(e).__name__}({e})"
    return ToolCallResult(tool_call_id=call.id, result=result)
