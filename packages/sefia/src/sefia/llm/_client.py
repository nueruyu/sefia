from abc import ABC, abstractmethod
from typing import Callable, Coroutine

from sefia.llm._messages import LLMCompletion, Message
from sefia.llm.step_decision import DecisionSpec, StepTool
from sefia.llm.streaming import OutputStreamCallback


class LLMClient(ABC):
    """
    Protocol for a client that interacts with a Large Language Model.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[StepTool] | None = None,
        decision_spec: DecisionSpec | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
        output_callback: OutputStreamCallback | None = None,
        reasoning_callback: (
            Callable[[str], Coroutine[None, None, None]] | None
        ) = None,
    ) -> LLMCompletion:
        """
        Sends a request to the LLM and returns a normalized completion.
        If stream_callback is provided, it will be called for each content token
        received. If reasoning_callback is provided, it will be called for each
        reasoning (thinking) token a reasoning model emits, which arrives before
        the completion content.

        Raises ``sefia.llm.exceptions.LLMCompletionDecodingError`` when the provider
        returned data but the client cannot represent it as ``LLMCompletion``.
        The exception must carry the partial completion so the inference strategy can
        treat the failure as repairable.
        """
        ...
