import inspect
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, TypeVar

from glyff import engrave
from pydantic import Field
from sefia import preview
from sefia.streaming import ArgStream, StringDelta

from .._session_state import get_call_state_store
from ..exceptions import NeedsInput

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]


@dataclass(frozen=True)
class InputRequest:
    """A request for external input."""

    interaction_id: str
    prompt: str


@dataclass(frozen=True)
class InputResult:
    """A completed external input interaction."""

    interaction_id: str
    prompt: str
    value: str


InputProvider = Callable[[InputRequest], MaybeAwaitable[str | None]]
InputRequestCallback = Callable[[InputRequest], MaybeAwaitable[None]]
InputCompleteCallback = Callable[[InputResult], MaybeAwaitable[None]]
InputPromptDeltaCallback = Callable[[str], MaybeAwaitable[None]]


@dataclass
class _InputCallState:
    """Internal state for a single get_input tool call."""

    interaction_id: str | None = None


async def _maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


async def _no_input(_: InputRequest) -> str | None:
    return None


class InputTool:
    def __init__(
        self,
        get_input: InputProvider = _no_input,
        on_request: InputRequestCallback | None = None,
        on_complete: InputCompleteCallback | None = None,
        on_prompt_delta: InputPromptDeltaCallback | None = None,
    ) -> None:
        self._get_input = get_input
        self._on_request = on_request
        self._on_complete = on_complete
        self._on_prompt_delta = on_prompt_delta

    async def _notify_request(self, request: InputRequest) -> None:
        if self._on_request is not None:
            await _maybe_await(self._on_request(request))

    async def _notify_complete(self, result: InputResult) -> None:
        if self._on_complete is not None:
            await _maybe_await(self._on_complete(result))

    async def _notify_prompt_delta(self, text: str) -> None:
        if self._on_prompt_delta is not None:
            await _maybe_await(self._on_prompt_delta(text))

    @engrave
    async def get_input(
        self,
        prompt: Annotated[str, Field(min_length=1)],
    ) -> str:
        """
        Request external input for ``prompt`` and return the provided value.

        The prompt is emitted to the configured input callbacks. If no input
        is immediately available, the current session is interrupted until
        it is provided.
        """
        call_store = get_call_state_store("internal_state", _InputCallState)
        call_state = await call_store.ensure()

        if call_state.interaction_id is None:
            call_state.interaction_id = str(uuid.uuid4())
            await call_store.save(call_state)

        request = InputRequest(
            interaction_id=call_state.interaction_id,
            prompt=prompt,
        )
        value = await _maybe_await(self._get_input(request))
        if value is not None:
            await self._notify_complete(
                InputResult(
                    interaction_id=request.interaction_id,
                    prompt=prompt,
                    value=value,
                )
            )
            return value

        await self._notify_request(request)
        raise NeedsInput(prompt, interaction_id=request.interaction_id)

    @preview(get_input)
    async def _stream_get_input(self, events: ArgStream) -> None:
        async for event in events:
            if isinstance(event, StringDelta) and event.name == "prompt":
                await self._notify_prompt_delta(event.text)
