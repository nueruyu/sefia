from collections.abc import Callable
from typing import Annotated

from pydantic import Field
from sefia import current_tool_call_id_for, preview
from sefia.streaming import ArgStream, StringDelta

from .._async import MaybeAwaitable, maybe_await
from .._glyff import GLYFF_DOMAIN
from .._input import (
    InputCompleteCallback,
    InputProvider,
    InputRequest,
    InputRequestCallback,
    InputResult,
    no_input,
    request_input,
)

InputPromptDeltaCallback = Callable[[str, str], MaybeAwaitable[None]]

__all__ = [
    "Input",
    "InputRequest",
    "InputResult",
    "InputProvider",
    "InputRequestCallback",
    "InputCompleteCallback",
    "InputPromptDeltaCallback",
]


class Input:
    def __init__(
        self,
        get_input: InputProvider = no_input,
        on_request: InputRequestCallback | None = None,
        on_complete: InputCompleteCallback | None = None,
        on_prompt_delta: InputPromptDeltaCallback | None = None,
    ) -> None:
        self._get_input = get_input
        self._on_request = on_request
        self._on_complete = on_complete
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

        ``prompt`` is an optional question to elicit the input; when given it
        is emitted to the configured input callbacks. Omit it for a bare
        ask-and-wait (use ``Output.send_output`` for non-blocking
        narration). If no input is immediately available, the current session
        is interrupted until it is provided.
        """
        interaction_id = current_tool_call_id_for(self.get_input)
        if interaction_id is None:
            raise RuntimeError(
                "Input.get_input() must be invoked as a dispatched tool."
            )
        request = InputRequest(
            interaction_id=interaction_id,
            prompt=prompt or "",
        )
        return await request_input(
            request, self._get_input, self._on_request, self._on_complete
        )

    @preview(get_input)
    async def _stream_get_input(self, tool_call_id: str, events: ArgStream) -> None:
        async for event in events:
            if isinstance(event, StringDelta) and event.name == "prompt":
                await self._notify_prompt_delta(tool_call_id, event.text)
