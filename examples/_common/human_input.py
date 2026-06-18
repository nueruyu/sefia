import inspect
from collections.abc import Awaitable, Callable
from typing import TypeVar

from sefia import get_context
from sefios.tools import HumanInputRequest, HumanInputResult, HumanInputTool

_PENDING_HUMAN_INPUTS_KEY = "pending_human_inputs"
# Inputs supplied before any request is pending are queued here (FIFO) so that
# providing several seed inputs in a row does not silently overwrite earlier
# ones.
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

    async def receive_input(
        self,
        input_text: str,
        *,
        reply_to: str | None = None,
    ) -> None:
        """Store CLI input as an answer for a pending or upcoming interaction."""
        interaction_id = await self._resolve_answer_target(reply_to)
        session_store = get_context().session_store
        if interaction_id is None:
            queue = await session_store.get(_UNCLAIMED_HUMAN_INPUTS_KEY, list) or []
            queue.append(input_text)
            await session_store.set(_UNCLAIMED_HUMAN_INPUTS_KEY, queue, list)
            return

        await session_store.set(self._answer_key(interaction_id), input_text, str)

    async def get_pending_requests(self) -> dict[str, dict]:
        session_store = get_context().session_store
        pending = await session_store.get(_PENDING_HUMAN_INPUTS_KEY, dict)
        return pending or {}

    async def get_answer(self, request: HumanInputRequest) -> str | None:
        session_store = get_context().session_store
        answer = await session_store.get(self._answer_key(request.interaction_id), str)
        if answer is not None:
            return answer

        # Only fall back to a queued, unclaimed input when this request is
        # unambiguously the one awaiting an answer. If any other distinct request
        # is pending, an unclaimed input must not be silently attributed here —
        # the caller is expected to answer it explicitly (e.g. via --reply-to).
        pending = await self.get_pending_requests()
        if any(
            interaction_id != request.interaction_id for interaction_id in pending
        ):
            return None

        queue = await session_store.get(_UNCLAIMED_HUMAN_INPUTS_KEY, list)
        if not queue:
            return None

        next_answer = queue.pop(0)
        if queue:
            await session_store.set(_UNCLAIMED_HUMAN_INPUTS_KEY, queue, list)
        else:
            await session_store.delete(_UNCLAIMED_HUMAN_INPUTS_KEY)
        return next_answer

    async def handle_request(self, request: HumanInputRequest) -> None:
        session_store = get_context().session_store
        pending = await self.get_pending_requests()
        pending[request.interaction_id] = {
            "id": request.interaction_id,
            "question": request.question,
        }
        await session_store.set(_PENDING_HUMAN_INPUTS_KEY, pending, dict)

        if self._on_request is not None:
            await _maybe_await(self._on_request(request))

    async def handle_complete(self, result: HumanInputResult) -> None:
        session_store = get_context().session_store
        # Intentionally keep the stored answer in place. The tool memoizes its
        # result via @engrave only after get_human_input returns, so deleting the
        # answer here (before the return is durably recorded) would leave a window
        # where a crash loses the answer and the question is re-asked on resume.
        # Keeping the answer keyed by interaction_id makes a re-run idempotent;
        # the leftover value is inert and namespaced to a single interaction.
        pending = await self.get_pending_requests()
        pending.pop(result.interaction_id, None)
        if pending:
            await session_store.set(_PENDING_HUMAN_INPUTS_KEY, pending, dict)
        else:
            await session_store.delete(_PENDING_HUMAN_INPUTS_KEY)

    async def _resolve_answer_target(self, reply_to: str | None) -> str | None:
        pending = await self.get_pending_requests()
        if reply_to is not None:
            if reply_to not in pending:
                raise UnknownHumanInputError(reply_to)
            return reply_to

        if not pending:
            return None

        if len(pending) == 1:
            return next(iter(pending))

        raise AmbiguousHumanInputError(sorted(pending))

    @staticmethod
    def _answer_key(interaction_id: str) -> str:
        return f"human_input__{interaction_id}"


async def _maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value
