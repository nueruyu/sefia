from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .._tool_system import Tool


class DecisionMode(Enum):
    """The shape of decision a model is allowed to return."""

    TOOL_ONLY = "tool_only"
    TOOL_ENABLED = "tool_enabled"
    OUTPUT_ONLY = "output_only"


@dataclass(frozen=True)
class DecisionModelSpec:
    """Requested decision contract for an inference strategy."""

    name: str
    output_type: Any
    tools: list[Tool]
    mode: DecisionMode

    def __post_init__(self) -> None:
        if self.mode in (DecisionMode.TOOL_ONLY, DecisionMode.TOOL_ENABLED):
            if not self.tools:
                raise ValueError(
                    f"{self.mode.value} decisions require at least one tool."
                )
        elif self.mode is DecisionMode.OUTPUT_ONLY:
            if self.tools:
                raise ValueError("output_only decisions cannot include tools.")
        else:
            raise ValueError(f"Unsupported decision mode: {self.mode!r}")

    @classmethod
    def tool_only(
        cls,
        *,
        name: str,
        output_type: Any,
        tools: list[Tool],
    ) -> "DecisionModelSpec":
        return cls(
            name=name,
            output_type=output_type,
            tools=tools,
            mode=DecisionMode.TOOL_ONLY,
        )

    @classmethod
    def tool_enabled(
        cls,
        *,
        name: str,
        output_type: Any,
        tools: list[Tool],
    ) -> "DecisionModelSpec":
        return cls(
            name=name,
            output_type=output_type,
            tools=tools,
            mode=DecisionMode.TOOL_ENABLED,
        )

    @classmethod
    def output_only(
        cls,
        *,
        name: str,
        output_type: Any,
    ) -> "DecisionModelSpec":
        return cls(
            name=name,
            output_type=output_type,
            tools=[],
            mode=DecisionMode.OUTPUT_ONLY,
        )


@dataclass
class DecisionToolCall:
    """A validated tool call decision."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalAnswerLLMDecision:
    """A validated decision to return a final answer."""

    final_answer: Any


@dataclass(frozen=True)
class ToolCallsLLMDecision:
    """A validated decision to call one or more tools."""

    tool_calls: list[DecisionToolCall]


LLMDecision = FinalAnswerLLMDecision | ToolCallsLLMDecision


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
