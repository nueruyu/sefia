from abc import ABC, abstractmethod
from typing import Callable, Coroutine

from sefia.llm._messages import LLMResponse, Message
from sefia.llm.step_decision import DecisionSpec, StepTool
from sefia.llm.streaming import OutputStreamCallback


class LLMResponseDecodingError(ValueError):
    """The provider returned a response the client could not represent safely.

    ``LLMClient`` implementations must raise this exception, rather than a generic
    decoding exception, when a received response is malformed or cannot be mapped to
    ``LLMResponse``. Transports use it to preserve the partial response and route the
    failure through the inference repair flow.
    """

    def __init__(self, response: LLMResponse, reason: str) -> None:
        super().__init__(reason)
        self.response = response


class LLMClient(ABC):
    """
    Protocol for a client that interacts with a Large Language Model.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[StepTool] | None = None,
        decision_model: DecisionSpec | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
        output_callback: OutputStreamCallback | None = None,
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

        Raises ``LLMResponseDecodingError`` when the provider returned a response but
        the client cannot represent it as ``LLMResponse``. The exception must carry
        the partial response so a decision transport can treat the failure as
        repairable.
        """
        ...
