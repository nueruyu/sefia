import inspect
from collections.abc import Awaitable, Callable
from typing import TypeVar

from sefia import get_context
from sefios.tools import HumanInputRequest, HumanInputResult, HumanInputTool

_PENDING_HUMAN_INTERACTION_KEY = "pending_human_interaction"

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]
HumanInputRequestHandler = Callable[[HumanInputRequest], MaybeAwaitable[None]]


class CLIHumanInputAdapter:
    """Connects HumanInputTool callbacks to the CLI session protocol."""

    def __init__(self, *, on_request: HumanInputRequestHandler | None = None):
        self._on_request = on_request

    def create_tool(self) -> HumanInputTool:
        return HumanInputTool(
            get_answer=self.get_answer,
            on_request=self.handle_request,
            on_complete=self.handle_complete,
        )

    async def receive_input(self, input_text: str, *, is_new: bool) -> None:
        """Stores the CLI input as the answer for a pending human interaction."""
        if is_new:
            return

        pending = await self.get_pending_request()
        if pending is None:
            return

        interaction_id = pending["id"]
        session_store = get_context().session_store
        await session_store.set(self._answer_key(interaction_id), input_text, str)

    async def get_pending_request(self) -> dict | None:
        session_store = get_context().session_store
        return await session_store.get(_PENDING_HUMAN_INTERACTION_KEY, dict)

    async def get_answer(self, request: HumanInputRequest) -> str | None:
        session_store = get_context().session_store
        return await session_store.get(self._answer_key(request.interaction_id), str)

    async def handle_request(self, request: HumanInputRequest) -> None:
        session_store = get_context().session_store
        await session_store.set(
            _PENDING_HUMAN_INTERACTION_KEY,
            {"id": request.interaction_id, "question": request.question},
            dict,
        )
        if self._on_request is not None:
            await _maybe_await(self._on_request(request))

    async def handle_complete(self, result: HumanInputResult) -> None:
        session_store = get_context().session_store
        await session_store.delete(self._answer_key(result.interaction_id))
        await session_store.delete(_PENDING_HUMAN_INTERACTION_KEY)

    @staticmethod
    def _answer_key(interaction_id: str) -> str:
        return f"human_input__{interaction_id}"


async def _maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value
