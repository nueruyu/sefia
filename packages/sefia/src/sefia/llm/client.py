from abc import ABC, abstractmethod

from sefia.llm.messages import LLMResponse, Message


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
    ) -> LLMResponse:
        """Sends a completion request to the LLM and gets a response."""
        ...
