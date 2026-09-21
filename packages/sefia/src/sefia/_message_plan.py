from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from .inference import FunctionInfo

if TYPE_CHECKING:
    from .llm._messages import Message


@dataclass(frozen=True)
class TaskPrompt:
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class MessagePlan:
    parts: tuple[Message | TaskPrompt, ...]

    def __post_init__(self) -> None:
        count = sum(isinstance(part, TaskPrompt) for part in self.parts)
        if count != 1:
            raise ValueError(
                f"MessagePlan must contain exactly one TaskPrompt; found {count}."
            )

    @classmethod
    def default(cls, function: FunctionInfo) -> MessagePlan:
        return cls(parts=(TaskPrompt(arguments=dict(function.prompt_arguments)),))
