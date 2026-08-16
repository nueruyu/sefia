from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine

from sefia.llm._messages import LLMResponse, Message
from sefia.llm.structured_output import StructuredOutputSchema
from sefia.llm.streaming import StructuredOutputCallback


class LLMClient(ABC):
    """
    Protocol for a client that interacts with a Large Language Model.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        output_schema: StructuredOutputSchema | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
        structured_output_callback: StructuredOutputCallback | None = None,
        reasoning_callback: (
            Callable[[str], Coroutine[None, None, None]] | None
        ) = None,
    ) -> LLMResponse:
        """
        Sends a completion request to the LLM and gets a response.
        If stream_callback is provided, it will be called for each content token
        received. If reasoning_callback is provided, it will be called for each
        reasoning (thinking) token a reasoning model emits, which arrives before
        the response content.
        """
        ...
