from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Never, Protocol

from .._tool_system import ToolEntry
from ..inference import StepDecision
from .structured_output import StructuredOutputSchema, StructuredValue


class StepDecisionMode(Enum):
    TOOLS_REQUIRED = "tools_required"
    TOOLS_OR_RESULT = "tools_or_result"
    RESULT_ONLY = "result_only"


@dataclass(frozen=True)
class StepDecisionSpec:
    name: str
    output_type: Any
    tools: list[ToolEntry]
    mode: StepDecisionMode

    def __post_init__(self) -> None:
        if self.mode in (
            StepDecisionMode.TOOLS_REQUIRED,
            StepDecisionMode.TOOLS_OR_RESULT,
        ):
            if not self.tools:
                raise ValueError(
                    f"{self.mode.value} decisions require at least one tool."
                )
        elif self.mode is StepDecisionMode.RESULT_ONLY:
            if self.tools:
                raise ValueError("result_only cannot include tools.")
        else:
            raise ValueError(f"Unsupported step decision mode: {self.mode!r}")

    @classmethod
    def tools_required(
        cls, *, name: str, output_type: Any, tools: list[ToolEntry]
    ) -> "StepDecisionSpec":
        return cls(name, output_type, tools, StepDecisionMode.TOOLS_REQUIRED)

    @classmethod
    def tools_or_result(
        cls, *, name: str, output_type: Any, tools: list[ToolEntry]
    ) -> "StepDecisionSpec":
        return cls(name, output_type, tools, StepDecisionMode.TOOLS_OR_RESULT)

    @classmethod
    def result_only(cls, *, name: str, output_type: Any) -> "StepDecisionSpec":
        return cls(name, output_type, [], StepDecisionMode.RESULT_ONLY)

    @classmethod
    def for_inference(
        cls, *, name: str, output_type: Any, tools: list[ToolEntry]
    ) -> "StepDecisionSpec":
        if output_type is Never:
            if not tools:
                raise ValueError(
                    "An @infer function returning Never must have tools available, "
                    "otherwise the inference loop can never make progress."
                )
            return cls.tools_required(name=name, output_type=output_type, tools=tools)
        if tools:
            return cls.tools_or_result(name=name, output_type=output_type, tools=tools)
        return cls.result_only(name=name, output_type=output_type)


class ToolCallIdSource(Protocol):
    def get_or_create(self, index: int) -> str: ...


class StepDecisionSchema(ABC):
    @property
    @abstractmethod
    def structured_output(self) -> StructuredOutputSchema: ...

    @abstractmethod
    def validate(
        self, value: StructuredValue, tool_call_ids: ToolCallIdSource | None
    ) -> StepDecision: ...


class StepDecisionSchemaFactory(ABC):
    @abstractmethod
    def create(self, spec: StepDecisionSpec) -> StepDecisionSchema: ...


__all__ = [
    "StepDecisionMode",
    "StepDecisionSchema",
    "StepDecisionSchemaFactory",
    "StepDecisionSpec",
    "ToolCallIdSource",
]
