from collections.abc import Callable
from typing import Annotated

from pydantic import Field
from sefia import current_tool_call_id_for, preview
from sefia.streaming import ArgStream, StringDelta

from .._async import MaybeAwaitable, maybe_await
from .._glyff import GLYFF_DOMAIN
from ..interactions import require_interaction

InputPromptDeltaCallback = Callable[[str, str], MaybeAwaitable[None]]

__all__ = ["Input", "InputPromptDeltaCallback"]


class Input:
    def __init__(
        self,
        on_prompt_delta: InputPromptDeltaCallback | None = None,
    ) -> None:
        self._on_prompt_delta = on_prompt_delta

    async def _notify_prompt_delta(self, interaction_id: str, text: str) -> None:
        if self._on_prompt_delta is not None:
            await maybe_await(self._on_prompt_delta(interaction_id, text))

    @GLYFF_DOMAIN.engrave(name="tools.input.get_input")
    async def get_input(
        self,
        prompt: Annotated[str, Field(min_length=1)] | None = None,
    ) -> str:
        """
        Request external input and return the provided value.

        ``prompt`` is included in the interaction request. Omit it for a bare
        ask-and-wait; use ``Output.send_output`` for non-blocking narration.
        An unresolved interaction pauses the current execution.
        """
        interaction_id = current_tool_call_id_for(self.get_input)
        if interaction_id is None:
            raise RuntimeError(
                "Input.get_input() must be invoked as a dispatched tool."
            )
        return await require_interaction(
            interaction_id, {"type": "input", "prompt": prompt or ""}, str
        )

    @preview(get_input)
    async def _stream_get_input(self, tool_call_id: str, events: ArgStream) -> None:
        async for event in events:
            if isinstance(event, StringDelta) and event.name == "prompt":
                await self._notify_prompt_delta(tool_call_id, event.text)
