from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

from litellm.types.utils import (  # pyright: ignore[reportMissingTypeStubs]
    Delta,
    ModelResponseStream,
    StreamingChoices,
)

from sefia.llm import Message
from sefia_litellm import LiteLLMClient


async def _completion_stream(
    *chunks: ModelResponseStream,
) -> AsyncIterator[ModelResponseStream]:
    for chunk in chunks:
        yield chunk


async def test_client_reconstructs_real_litellm_stream_chunks(
    mock_acompletion: AsyncMock,
) -> None:
    chunks = [
        ModelResponseStream(
            id="completion-1",
            model="gpt-4o",
            object="chat.completion.chunk",
            choices=[StreamingChoices(index=0, delta=Delta(content="hel"))],
        ),
        ModelResponseStream(
            id="completion-1",
            model="gpt-4o",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="lo"),
                    finish_reason="stop",
                )
            ],
        ),
    ]
    mock_acompletion.return_value = _completion_stream(*chunks)
    callback = AsyncMock()

    response = await LiteLLMClient(model="gpt-4o").complete(
        [Message(role="user", content="Hello")], stream_callback=callback
    )

    assert response.content == "hello"
    assert [call.args[0] for call in callback.await_args_list] == ["hel", "lo"]
