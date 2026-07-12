from abc import ABC, abstractmethod
from typing import Callable, Coroutine

from sefia.llm._messages import LLMResponse, Message


class LLMClient(ABC):
    """
    Protocol for a client that interacts with a Large Language Model.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        output_schema: dict | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
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
