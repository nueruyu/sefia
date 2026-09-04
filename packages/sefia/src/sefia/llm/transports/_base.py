from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ...inference import FunctionInfo, HistoryItem
from .._client import LLMClient
from .._messages import LLMResponse
from .._prompt_renderer import DecisionPrompt, PromptRenderer, RejectedDecision
from ..llm_output import LLMOutput
from ..step_decision import DecisionSpec, StepTool
from ..streaming import OutputStreamEvent


class DecisionObserver(ABC):
    @abstractmethod
    async def before_request(self, prompt: str) -> None: ...

    @abstractmethod
    async def response_text(self, text: str) -> None: ...

    @abstractmethod
    async def reasoning_text(self, text: str) -> None: ...

    @abstractmethod
    async def output(self, event: OutputStreamEvent) -> None: ...


@dataclass(frozen=True)
class DecisionRequest:
    function: FunctionInfo
    spec: DecisionSpec
    history: tuple[HistoryItem, ...]
    rejected: RejectedDecision | None = None

    def to_prompt(
        self,
        response_instructions: str,
        *,
        tools: tuple[StepTool, ...],
        history: tuple[HistoryItem, ...],
    ) -> DecisionPrompt:
        return DecisionPrompt(
            function=self.function,
            tools=tools,
            history=history,
            response_instructions=response_instructions,
            rejected=self.rejected,
        )


@dataclass(frozen=True)
class DecisionResponse:
    output: LLMOutput
    raw: LLMResponse


class DecisionDecodingError(ValueError):
    def __init__(self, response: LLMResponse, reason: str) -> None:
        super().__init__(reason)
        self.response = response


class DecisionTransport(ABC):
    """Requests one decision and decodes the response."""

    @abstractmethod
    async def request_decision(
        self,
        client: LLMClient,
        prompt_renderer: PromptRenderer,
        request: DecisionRequest,
        observer: DecisionObserver,
        stream: bool,
    ) -> DecisionResponse: ...
