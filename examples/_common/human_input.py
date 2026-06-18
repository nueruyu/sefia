import inspect
from collections.abc import Awaitable, Callable
from typing import TypeVar

from sefia import get_context
from sefios.tools import HumanInputRequest, HumanInputResult, HumanInputTool

_PENDING_HUMAN_INPUTS_KEY = "pending_human_inputs"
_UNCLAIMED_HUMAN_INPUTS_KEY = "unclaimed_human_inputs"

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]
HumanInputRequestHandler = Callable[[HumanInputRequest], MaybeAwaitable[None]]


class AmbiguousHumanInputError(Exception):
    """Raised when multiple pending human inputs need an explicit reply target."""

    def __init__(self, interaction_ids: list[str]):
        super().__init__(
            "Multiple pending human inputs exist. Specify one with --reply-to: "
            + ", ".join(interaction_ids)
        )
        self.interaction_ids = interaction_ids


class UnknownHumanInputError(Exception):
    """Raised when a CLI input targets an unknown pending human input."""

    def __init__(self, interaction_id: str):
        super().__init__(f"Unknown pending human input: {interaction_id}")
        self.interaction_id = interaction_id


class HumanInputSessionStore:
    """Small persistence wrapper for human-input state in the active session."""

    async def pending_requests(self) -> dict[str, dict]:
        session_store = get_context().session_store
        pending = await session_store.get(_PENDING_HUMAN_INPUTS_KEY, dict) or {}
        if not pending:
            return {}

        unanswered = {}
        for interaction_id, request in pending.items():
            answer = await session_store.get(self._answer_key(interaction_id), str)
            if answer is None:
                unanswered[interaction_id] = request

        await self.save_pending_requests(unanswered)
        return unanswered

    async def save_pending_requests(self, pending: dict[str, dict]) -> None:
        session_store = get_context().session_store
        if pending:
            await session_store.set(_PENDING_HUMAN_INPUTS_KEY, pending, dict)
            return

        await session_store.delete(_PENDING_HUMAN_INPUTS_KEY)

    async def get_answer(self, interaction_id: str) -> str | None:
        session_store = get_context().session_store
        return await session_store.get(self._answer_key(interaction_id), str)

    async def set_answer(self, interaction_id: str, answer: str) -> None:
        session_store = get_context().session_store
        await session_store.set(self._answer_key(interaction_id), answer, str)

    async def queue_input(self, input_text: str) -> None:
        session_store = get_context().session_store
        queue = await session_store.get(_UNCLAIMED_HUMAN_INPUTS_KEY, list) or []
        queue.append(input_text)
        await session_store.set(_UNCLAIMED_HUMAN_INPUTS_KEY, queue, list)

    async def pop_queued_input(self) -> str | None:
        session_store = get_context().session_store
        queue = await session_store.get(_UNCLAIMED_HUMAN_INPUTS_KEY, list)
        if not queue:
            return None

        next_input = queue.pop(0)
        if queue:
            await session_store.set(_UNCLAIMED_HUMAN_INPUTS_KEY, queue, list)
        else:
            await session_store.delete(_UNCLAIMED_HUMAN_INPUTS_KEY)
        return next_input

    @staticmethod
    def _answer_key(interaction_id: str) -> str:
        return f"human_input__{interaction_id}"


class CLIHumanInputReceiver:
    """Accepts CLI input and stores it for pending or future human-input requests."""

    def __init__(self, store: HumanInputSessionStore):
        self._store = store

    async def receive_input(
        self,
        input_text: str,
        *,
        reply_to: str | None = None,
    ) -> None:
        pending = await self._store.pending_requests()

        if reply_to is not None:
            if reply_to not in pending:
                raise UnknownHumanInputError(reply_to)
            await self._store.set_answer(reply_to, input_text)
            return

        if len(pending) == 1:
            await self._store.set_answer(next(iter(pending)), input_text)
            return

        if len(pending) > 1:
            raise AmbiguousHumanInputError(sorted(pending))

        await self._store.queue_input(input_text)


class CLIHumanInputAdapter:
    """Connects HumanInputTool callbacks to CLI reporting and session state."""

    def __init__(
        self,
        *,
        store: HumanInputSessionStore | None = None,
        on_request: HumanInputRequestHandler | None = None,
    ):
        self.store = store or HumanInputSessionStore()
        self._on_request = on_request

    def create_tool(self) -> HumanInputTool:
        return HumanInputTool(
            get_answer=self._get_answer,
            on_request=self._handle_request,
            on_complete=self._handle_complete,
        )

    async def _get_answer(self, request: HumanInputRequest) -> str | None:
        answer = await self.store.get_answer(request.interaction_id)
        if answer is not None:
            return answer

        pending = await self.store.pending_requests()
        if any(interaction_id != request.interaction_id for interaction_id in pending):
            return None

        return await self.store.pop_queued_input()

    async def _handle_request(self, request: HumanInputRequest) -> None:
        pending = await self.store.pending_requests()
        pending[request.interaction_id] = {
            "id": request.interaction_id,
            "question": request.question,
        }
        await self.store.save_pending_requests(pending)
        if self._on_request is not None:
            await _maybe_await(self._on_request(request))

    async def _handle_complete(self, result: HumanInputResult) -> None:
        pending = await self.store.pending_requests()
        pending.pop(result.interaction_id, None)
        await self.store.save_pending_requests(pending)


async def _maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value
