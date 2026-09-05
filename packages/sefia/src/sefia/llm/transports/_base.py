from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ...inference import FunctionInfo, HistoryItem
from .._client import LLMClient
from .._messages import LLMCompletion
from .._prompt_renderer import DecisionPrompt, PromptRenderer, RejectedDecision
from ..structured_data import StructuredData
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
class DecodedDecision:
    """Decision data decoded by a transport, before semantic validation."""

    decision_data: StructuredData
    completion: LLMCompletion


class DecisionTransport(ABC):
    """Requests one completion and decodes its decision protocol."""

    @abstractmethod
    async def request_decision(
        self,
        client: LLMClient,
        prompt_renderer: PromptRenderer,
        request: DecisionRequest,
        observer: DecisionObserver,
        stream: bool,
    ) -> DecodedDecision: ...
