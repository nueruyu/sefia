import inspect
from typing import Awaitable, Callable, TypeVar

from sefia._interfaces.middleware import StepContext, StepMiddleware
from sefia.inference import InferenceDecision, ToolCallDecision, ToolCallRequest
from sefios.tools.human import HumanInputTool

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]
QuestionComposer = Callable[[list[str]], MaybeAwaitable[str]]


async def _maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


def _join_questions(questions: list[str]) -> str:
    return "\n".join(questions)


async def _compose_human_input_calls(
    decision: ToolCallDecision,
    human_input_tool_names: set[str],
    compose_questions: QuestionComposer,
) -> ToolCallDecision:
    """Compose batched human-input calls in one decision into one prompt."""
    human_calls = [
        call for call in decision.calls if call.name in human_input_tool_names
    ]
    if len(human_calls) <= 1:
        return decision

    questions: list[str] = []
    for call in human_calls:
        question = call.arguments.get("question")
        if not isinstance(question, str) or not question:
            return decision
        questions.append(question)

    composed_question = await _maybe_await(compose_questions(questions))
    first_human_call = human_calls[0]
    composed_call = ToolCallRequest(
        id=first_human_call.id,
        name=first_human_call.name,
        arguments={
            **first_human_call.arguments,
            "question": composed_question,
        },
    )

    calls: list[ToolCallRequest] = []
    composed_inserted = False
    for call in decision.calls:
        if call.name not in human_input_tool_names:
            calls.append(call)
            continue
        if not composed_inserted:
            calls.append(composed_call)
            composed_inserted = True

    return ToolCallDecision(calls=calls)


class HumanInputCallComposer(StepMiddleware):
    """
    Composes multiple human-input requests emitted in the same inference step.

    The middleware only rewrites one ``ToolCallDecision`` batch at a time. Later
    resumed steps still produce their own independent human-input requests.
    """

    def __init__(self, compose_questions: QuestionComposer = _join_questions) -> None:
        self._compose_questions = compose_questions

    async def wrap(
        self,
        ctx: StepContext,
        nxt: Callable[[], Awaitable[InferenceDecision]],
    ) -> InferenceDecision:
        decision = await nxt()
        if not isinstance(decision, ToolCallDecision):
            return decision

        human_input_tool_names = {
            tool.name
            for tool in ctx.tool_registry.get_by_function(
                HumanInputTool.get_human_input
            )
        }
        if not human_input_tool_names:
            return decision

        return await _compose_human_input_calls(
            decision,
            human_input_tool_names,
            self._compose_questions,
        )
