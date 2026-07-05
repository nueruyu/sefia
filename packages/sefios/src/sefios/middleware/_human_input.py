from typing import Awaitable, Callable

from sefia._interfaces.middleware import StepContext, StepMiddleware
from sefia.inference import InferenceDecision, ToolCallDecision, ToolCallRequest
from sefios.tools.human import HUMAN_INPUT_TOOL_NAME


def compose_human_input_calls(decision: ToolCallDecision) -> ToolCallDecision:
    """Compose batched human-input calls in one decision into one prompt."""
    human_calls = [
        call for call in decision.calls if call.name == HUMAN_INPUT_TOOL_NAME
    ]
    if len(human_calls) <= 1:
        return decision

    questions: list[str] = []
    for call in human_calls:
        question = call.arguments.get("question")
        if not isinstance(question, str) or not question:
            return decision
        questions.append(question)

    first_human_call = human_calls[0]
    composed_call = ToolCallRequest(
        id=first_human_call.id,
        name=first_human_call.name,
        arguments={
            **first_human_call.arguments,
            "question": _compose_question(questions),
        },
    )

    calls: list[ToolCallRequest] = []
    composed_inserted = False
    for call in decision.calls:
        if call.name != HUMAN_INPUT_TOOL_NAME:
            calls.append(call)
            continue
        if not composed_inserted:
            calls.append(composed_call)
            composed_inserted = True

    return ToolCallDecision(calls=calls)


def _compose_question(questions: list[str]) -> str:
    lines = ["Please answer these together:"]
    lines.extend(f"{i}. {question}" for i, question in enumerate(questions, start=1))
    return "\n".join(lines)


class ComposeHumanInputStepMiddleware(StepMiddleware):
    """
    Composes multiple human-input requests emitted in the same inference step.

    The middleware only rewrites one ``ToolCallDecision`` batch at a time. Later
    resumed steps still produce their own independent human-input requests.
    """

    async def wrap(
        self,
        ctx: StepContext,
        nxt: Callable[[], Awaitable[InferenceDecision]],
    ) -> InferenceDecision:
        decision = await nxt()
        if not isinstance(decision, ToolCallDecision):
            return decision
        return compose_human_input_calls(decision)
