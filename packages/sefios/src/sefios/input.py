"""External input from deterministic application control flow."""

from ._glyff import GLYFF_DOMAIN
from ._input import InputRequest, InputResult, request_input
from ._input_channel import get_input_channel


@GLYFF_DOMAIN.engrave(name="require_input")
async def require_input(prompt: str, *, allow_queued: bool = False) -> str:
    """Pause for a reply using the active HTTP or CLI session's input channel.

    By default, messages queued before the request are not consumed. Set
    ``allow_queued`` for conversational input that may use an earlier message.
    The returned text is not an approval decision; validate it in application code.
    """
    channel = get_input_channel()

    async def provide(request: InputRequest) -> str | None:
        return await channel.provide_input(
            request.interaction_id, allow_queued=allow_queued
        )

    async def record(request: InputRequest) -> None:
        await channel.record_request(request.interaction_id, request.prompt)

    async def complete(result: InputResult) -> None:
        await channel.complete_request(result.interaction_id)

    return await request_input(prompt, provide, record, complete)
