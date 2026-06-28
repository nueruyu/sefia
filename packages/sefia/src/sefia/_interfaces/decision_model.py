from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class DecisionToolSpec:
    """A tool available to the decision model."""

    name: str
    function: Callable[..., Any]
    schema: dict


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
    tools: list[DecisionToolSpec]
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
        tools: list[DecisionToolSpec],
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
        tools: list[DecisionToolSpec],
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
    def build(self, spec: DecisionModelSpec) -> DecisionModel:
        """Build a decision model with the requested decision contract."""
        ...
