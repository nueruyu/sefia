from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

from ...inference import FunctionInfo
from .._client import LLMClient
from .._messages import LLMCompletion, Message, ToolCall
from .._prompt_renderer import PromptRenderer
from ..structured_data import StructuredData
from ..step_decision import DecisionSpec
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
class RejectedDecision:
    completion: LLMCompletion
    reason: str


@dataclass(frozen=True)
class DecisionToolCalls:
    calls: tuple[ToolCall, ...]


@dataclass(frozen=True)
class DecisionToolResult:
    tool_call_id: str
    result: StructuredData


DecisionHistoryItem: TypeAlias = DecisionToolCalls | DecisionToolResult


@dataclass(frozen=True)
class DecisionRequest:
    """Semantic input ready for decision-protocol presentation."""

    messages_before: tuple[Message, ...]
    function: FunctionInfo
    arguments: StructuredData
    messages_after: tuple[Message, ...]
    decision_spec: DecisionSpec
    history: tuple[DecisionHistoryItem, ...]
    rejected: RejectedDecision | None = None

    def with_rejection(self, rejected: RejectedDecision) -> "DecisionRequest":
        return DecisionRequest(
            messages_before=self.messages_before,
            function=self.function,
            arguments=self.arguments,
            messages_after=self.messages_after,
            decision_spec=self.decision_spec,
            history=self.history,
            rejected=rejected,
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
