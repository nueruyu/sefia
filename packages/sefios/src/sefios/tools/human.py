import inspect
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, TypeVar

from glyff import engrave
from pydantic import Field
from sefia import preview
from sefia.exceptions import NeedsInput
from sefia.streaming import ArgStream, StringDelta

from .._session_state import get_session_state

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]


@dataclass(frozen=True)
class HumanInputRequest:
    """A request for external human input."""

    interaction_id: str
    question: str


@dataclass(frozen=True)
class HumanInputResult:
    """A completed external human input interaction."""

    interaction_id: str
    question: str
    answer: str


HumanInputAnswerProvider = Callable[[HumanInputRequest], MaybeAwaitable[str | None]]
HumanInputRequestCallback = Callable[[HumanInputRequest], MaybeAwaitable[None]]
HumanInputCompleteCallback = Callable[[HumanInputResult], MaybeAwaitable[None]]
HumanInputQuestionDeltaCallback = Callable[[str], MaybeAwaitable[None]]


@dataclass
class _AskUserState:
    """Internal state for a single get_human_input tool call."""

    interaction_id: str | None = None


async def _maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


async def _no_answer(_: HumanInputRequest) -> str | None:
    return None


class HumanInputTool:
    def __init__(
        self,
        get_answer: HumanInputAnswerProvider = _no_answer,
        on_request: HumanInputRequestCallback | None = None,
        on_complete: HumanInputCompleteCallback | None = None,
        on_question_delta: HumanInputQuestionDeltaCallback | None = None,
    ) -> None:
        self._get_answer = get_answer
        self._on_request = on_request
        self._on_complete = on_complete
        self._on_question_delta = on_question_delta

    async def _notify_request(self, request: HumanInputRequest) -> None:
        if self._on_request is not None:
            await _maybe_await(self._on_request(request))

    async def _notify_complete(self, result: HumanInputResult) -> None:
        if self._on_complete is not None:
            await _maybe_await(self._on_complete(result))

    async def _notify_question_delta(self, text: str) -> None:
        if self._on_question_delta is not None:
            await _maybe_await(self._on_question_delta(text))

    @engrave
    async def get_human_input(
        self,
        question: Annotated[str, Field(min_length=1)],
    ) -> str:
        """
        Request external human input for ``question`` and return the answer.

        The question is emitted to the configured human-input callbacks. If no
        answer is immediately available, the current session is interrupted until
        input is provided.
        """
        call_store = get_session_state().get_call_state_store(
            "internal_state", _AskUserState
        )
        call_state = await call_store.ensure()

        if call_state.interaction_id is None:
            call_state.interaction_id = str(uuid.uuid4())
            await call_store.save(call_state)

        request = HumanInputRequest(
            interaction_id=call_state.interaction_id,
            question=question,
        )
        answer = await _maybe_await(self._get_answer(request))
        if answer is not None:
            await self._notify_complete(
                HumanInputResult(
                    interaction_id=request.interaction_id,
                    question=question,
                    answer=answer,
                )
            )
            return answer

        await self._notify_request(request)
        raise NeedsInput(question)

    @preview(get_human_input)
    async def _stream_get_human_input(self, events: ArgStream) -> None:
        async for event in events:
            if isinstance(event, StringDelta) and event.name == "question":
                await self._notify_question_delta(event.text)
