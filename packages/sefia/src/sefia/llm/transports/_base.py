from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from .._message_plan import MessagePlan
from ...inference import FunctionInfo, HistoryItem
from .._client import LLMClient
from .._messages import LLMCompletion, Message
from .._prompt_renderer import DecisionPrompt, PromptRenderer, RejectedDecision
from ..structured_data import StructuredData
from ..step_decision import DecisionSpec, StepTool
from ..streaming import OutputStreamEvent


class DecisionObserver(ABC):
    @abstractmethod
    async def before_request(self, messages: tuple[Message, ...]) -> None: ...

    @abstractmethod
    async def response_text(self, text: str) -> None: ...

    @abstractmethod
    async def reasoning_text(self, text: str) -> None: ...

    @abstractmethod
    async def output(self, event: OutputStreamEvent) -> None: ...


@dataclass(frozen=True)
class DecisionRequest:
    function: FunctionInfo
    message_plan: MessagePlan
    decision_spec: DecisionSpec
    history: tuple[HistoryItem, ...]
    rejected: RejectedDecision | None = None

    def to_prompt(
        self,
        *,
        arguments: Mapping[str, Any],
        tools: tuple[StepTool, ...],
    ) -> DecisionPrompt:
        return DecisionPrompt(
            function=self.function,
            arguments=arguments,
            tools=tools,
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
