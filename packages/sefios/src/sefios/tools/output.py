from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from pydantic import Field
from sefia import current_tool_call_id_for, preview
from sefia.streaming import ArgStream, StringDelta

from .._async import MaybeAwaitable, maybe_await
from .._glyff import engrave


@dataclass(frozen=True)
class OutputMessage:
    """A message the agent emitted to the human."""

    interaction_id: str
    message: str


OutputCallback = Callable[[OutputMessage], MaybeAwaitable[None]]
OutputMessageDeltaCallback = Callable[[str, str], MaybeAwaitable[None]]


class Output:
    """Emits agent-authored messages to the human without blocking on a reply.

    The sibling of :class:`Input`: ``get_input`` asks and waits,
    ``send_output`` narrates and returns. ``on_output`` is the emit hook (a CLI
    reporter or an SSE stream); ``on_message_delta`` receives the message
    token-by-token as the model streams the call.
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
            await maybe_await(self._on_output(message))

    async def _notify_message_delta(self, interaction_id: str, text: str) -> None:
        if self._on_message_delta is not None:
            await maybe_await(self._on_message_delta(interaction_id, text))

    @engrave("output")
    async def send_output(
        self,
        message: Annotated[str, Field(min_length=1)],
    ) -> str:
        """
        Emit ``message`` to the human and return immediately.

        Use this to narrate progress or send an assistant message that is not a
        question; unlike ``get_input`` it does not wait for a response. Being
        engraved, the emit fires exactly once even across a resume.
        """
        interaction_id = current_tool_call_id_for(self.send_output)
        if interaction_id is None:
            raise RuntimeError(
                "Output.send_output() must be invoked as a dispatched tool."
            )
        output = OutputMessage(
            interaction_id=interaction_id,
            message=message,
        )
        await self._notify_output(output)
        return "Message delivered to the user."

    @preview(send_output)
    async def _stream_send_output(self, tool_call_id: str, events: ArgStream) -> None:
        async for event in events:
            if isinstance(event, StringDelta) and event.name == "message":
                await self._notify_message_delta(tool_call_id, event.text)
