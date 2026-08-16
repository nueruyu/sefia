from typing import Awaitable, Callable

from sefia._interfaces.middleware import StepContext, StepMiddleware
from sefia.inference import StepDecision, ToolCallsDecision, ToolCallRequest
from sefios.tools.input import Input
from typing_extensions import final, override

from .._async import MaybeAwaitable, maybe_await

PromptComposer = Callable[[list[str]], MaybeAwaitable[str]]


def _join_prompts(prompts: list[str]) -> str:
    return "\n".join(prompts)


async def _compose_input_calls(
    decision: ToolCallsDecision,
    input_tool_names: set[str],
    compose_prompts: PromptComposer,
) -> ToolCallsDecision:
    """Compose batched input calls in one decision into one prompt."""
    input_calls = [call for call in decision.calls if call.name in input_tool_names]
    if len(input_calls) <= 1:
        return decision

    prompts: list[str] = []
    for call in input_calls:
        prompt = call.arguments.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            return decision
        prompts.append(prompt)

    composed_prompt = await maybe_await(compose_prompts(prompts))
    first_input_call = input_calls[0]
    composed_call = ToolCallRequest(
        id=first_input_call.id,
        name=first_input_call.name,
        arguments={
            **first_input_call.arguments,
            "prompt": composed_prompt,
        },
    )

    calls: list[ToolCallRequest] = []
    composed_inserted = False
    for call in decision.calls:
        if call.name not in input_tool_names:
            calls.append(call)
            continue
        if not composed_inserted:
            calls.append(composed_call)
            composed_inserted = True

    return ToolCallsDecision(calls=calls)


@final
class InputCallComposer(StepMiddleware):
    """
    Composes multiple input requests emitted in the same inference step.

    The middleware only rewrites one ``ToolCallsDecision`` batch at a time. Later
    resumed steps still produce their own independent input requests.
    """

    def __init__(self, compose_prompts: PromptComposer = _join_prompts) -> None:
        self._compose_prompts = compose_prompts

    @override
    async def wrap(
        self,
        ctx: StepContext,
        nxt: Callable[[], Awaitable[StepDecision]],
    ) -> StepDecision:
        decision = await nxt()
        if not isinstance(decision, ToolCallsDecision):
            return decision

        input_tool_names = {
            tool.name for tool in ctx.tool_registry.get_by_function(Input.get_input)
        }
        if not input_tool_names:
            return decision

        return await _compose_input_calls(
            decision,
            input_tool_names,
            self._compose_prompts,
        )
