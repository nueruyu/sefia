from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

from ..exceptions import InvalidInferenceResponseError
from ..inference import HistoryItem, ResultDecision, ToolCallsDecision
from ._messages import LLMResponse, Message
from ._tool_call_ids import ToolCallIdRegistry
from .step_decision import StepDecisionModel, StepDecisionSpec

JsonDefault = Callable[[Any], Any]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    parameters: dict[str, Any]
    description: str | None = None


class ToolCallTransport(ABC):
    @property
    @abstractmethod
    def supports_arg_streaming(self) -> bool: ...

    @abstractmethod
    def definitions(self, model: StepDecisionModel) -> list[ToolDefinition] | None: ...

    @abstractmethod
    def decision_model(self, model: StepDecisionModel) -> StepDecisionModel | None: ...

    @abstractmethod
    def prompt(self, spec: StepDecisionSpec, model: StepDecisionModel) -> str: ...

    @abstractmethod
    def render_history(
        self, history: Sequence[HistoryItem], json_default: JsonDefault | None
    ) -> list[Message]: ...

    @abstractmethod
    def decode(
        self,
        response: LLMResponse,
        model: StepDecisionModel,
        tool_call_ids: ToolCallIdRegistry,
    ) -> ToolCallsDecision | None: ...

    @abstractmethod
    def repair_messages(
        self, error: InvalidInferenceResponseError
    ) -> list[Message]: ...


class ResultTransport(ABC):
    @abstractmethod
    def definitions(self, model: StepDecisionModel) -> list[ToolDefinition] | None: ...

    @abstractmethod
    def decision_model(self, model: StepDecisionModel) -> StepDecisionModel | None: ...

    @abstractmethod
    def prompt(self, spec: StepDecisionSpec, model: StepDecisionModel) -> str: ...

    @abstractmethod
    def decode(
        self, response: LLMResponse, model: StepDecisionModel
    ) -> ResultDecision | None: ...

    @abstractmethod
    def repair_messages(
        self, error: InvalidInferenceResponseError
    ) -> list[Message]: ...


__all__ = ["ResultTransport", "ToolCallTransport", "ToolDefinition"]
