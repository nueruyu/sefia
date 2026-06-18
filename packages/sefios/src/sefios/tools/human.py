import inspect
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from glyff import engrave
from glyff.exceptions import YieldException
from sefia import get_context, tool

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


@dataclass
class HumanInputTool:
    get_answer: HumanInputAnswerProvider = _no_answer
    on_request: HumanInputRequestCallback | None = None
    on_complete: HumanInputCompleteCallback | None = None

    async def _notify_request(self, request: HumanInputRequest) -> None:
        if self.on_request is not None:
            await _maybe_await(self.on_request(request))

    async def _notify_complete(self, result: HumanInputResult) -> None:
        if self.on_complete is not None:
            await _maybe_await(self.on_complete(result))

    @tool
    @engrave
    async def get_human_input(self, question: str) -> str:
        """
        Asks the user a question and returns their answer.
        This tool interrupts the session to wait for user input.
        """
        ctx = get_context()
        call_store = ctx.get_call_state_store("internal_state", _AskUserState)
        call_state = await call_store.ensure()

        if call_state.interaction_id is None:
            call_state.interaction_id = str(uuid.uuid4())
            await call_store.save(call_state)

        request = HumanInputRequest(
            interaction_id=call_state.interaction_id,
            question=question,
        )
        answer = await _maybe_await(self.get_answer(request))
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
        raise YieldException()
