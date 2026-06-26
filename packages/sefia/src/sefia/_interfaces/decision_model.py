from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DecisionToolSpec:
    """A tool available to the decision model."""

    name: str
    function: Callable[..., Any]
    schema: dict


@dataclass
class DecisionToolCall:
    """A validated tool call decision."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMDecision:
    """The validated decision returned by an LLM response."""

    final_answer: Any = None
    tool_calls: list[DecisionToolCall] | None = None


class DecisionModel(ABC):
    """Schema and validation boundary for LLM decisions."""

    @abstractmethod
    def schema(self) -> dict:
        """Return the JSON schema presented to the LLM."""
        ...

    @abstractmethod
    def validate(self, data: Any) -> LLMDecision:
        """Validate raw response data and return a normalized decision."""
        ...


class DecisionModelBuilder(ABC):
    """Builds decision models for an inference strategy."""

    @abstractmethod
    def build(
        self,
        *,
        name: str,
        output_type: Any,
        tools: list[DecisionToolSpec],
        include_final_answer: bool,
        include_tool_calls: bool,
        final_answer_nullable: bool,
        tool_calls_nullable: bool,
    ) -> DecisionModel:
        """Build a decision model with the requested fields."""
        ...
