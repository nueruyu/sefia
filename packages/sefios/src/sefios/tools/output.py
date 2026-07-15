import inspect
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, TypeVar

from glyff import engrave
from pydantic import Field
from sefia import preview
from sefia.streaming import ArgStream, StringDelta

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]


@dataclass(frozen=True)
class OutputMessage:
    """A message the agent emitted to the human."""

    interaction_id: str
    message: str


OutputCallback = Callable[[OutputMessage], MaybeAwaitable[None]]
OutputMessageDeltaCallback = Callable[[str], MaybeAwaitable[None]]


async def _maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


class OutputTool:
    """Emits agent-authored messages to the human without blocking on a reply.

    The sibling of :class:`InputTool`: where ``get_input`` asks and waits,
    ``send_output`` narrates and returns. The single engraved method emits
    exactly once -- glyff replays a completed engraved call from its recorded
    result without re-running the body, so a resumed run does not re-emit
    (the same durability that keeps ``get_input``'s request from re-firing).

    ``on_output`` is the emit hook (e.g. a CLI reporter or an SSE stream);
    ``on_message_delta`` receives the message token-by-token as the model
    streams the call, mirroring ``InputTool``'s prompt deltas.
    """

    def __init__(
        self,
        on_output: OutputCallback | None = None,
        on_message_delta: OutputMessageDeltaCallback | None = None,
    ) -> None:
        self._on_output = on_output
        self._on_message_delta = on_message_delta

    async def _notify_output(self, message: OutputMessage) -> None:
        if self._on_output is not None:
            await _maybe_await(self._on_output(message))

    async def _notify_message_delta(self, text: str) -> None:
        if self._on_message_delta is not None:
            await _maybe_await(self._on_message_delta(text))

    @engrave
    async def send_output(
        self,
        message: Annotated[str, Field(min_length=1)],
    ) -> str:
        """
        Emit ``message`` to the human and return immediately.

        Use this to narrate progress or to send an assistant message that is
        not a question. Unlike ``get_input`` it does not wait for a response,
        so the run keeps going. The message is emitted to the configured
        output callbacks exactly once, even across a resume.
        """
        output = OutputMessage(
            interaction_id=str(uuid.uuid4()),
            message=message,
        )
        await self._notify_output(output)
        return "Message delivered to the user."

    @preview(send_output)
    async def _stream_send_output(self, events: ArgStream) -> None:
        async for event in events:
            if isinstance(event, StringDelta) and event.name == "message":
                await self._notify_message_delta(event.text)
