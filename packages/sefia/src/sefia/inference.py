import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass
class ToolCallRequest:
    """Represents a request to call a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallResult:
    """Represents the result of a single tool call."""

    tool_call_id: str
    result: Any


@dataclass
class ToolCallDecision:
    """A decision to call one or more tools."""

    calls: list[ToolCallRequest]


@dataclass
class ResultDecision:
    """A decision to return the inference result."""

    result: Any


InferenceDecision = ToolCallDecision | ResultDecision
HistoryItem = ToolCallDecision | ToolCallResult


class History(Protocol):
    """The run history as step middleware sees it: read plus rewrite."""

    @property
    def items(self) -> Sequence[HistoryItem]: ...

    @property
    def completed_steps(self) -> int: ...

    async def rewrite(self, items: Sequence[HistoryItem]) -> None: ...


@dataclass(frozen=True)
class FunctionInfo:
    """Encapsulates the function to be inferred and its call information."""

    qualname: str
    instructions: str
    bound_arguments: dict[str, Any]
    type_hints: dict[str, Any]
    return_type: Any
    args: tuple
    kwargs: dict

    @classmethod
    def create(cls, func: Callable, args: tuple, kwargs: dict) -> "FunctionInfo":
        """Create a FunctionInfo instance from a function and its arguments."""
        type_hints = inspect.get_annotations(func, eval_str=True)
        instructions = inspect.getdoc(func) or "Execute the requested task."
        qualname = func.__qualname__
        return_type = type_hints.get("return", Any)

        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        return cls(
            qualname=qualname,
            instructions=instructions,
            bound_arguments=bound_args.arguments,
            type_hints=type_hints,
            return_type=return_type,
            args=args,
            kwargs=kwargs,
        )

    @property
    def instance(self) -> Any | None:
        """Return the instance ('self') if the function is a method."""
        return self.bound_arguments.get("self")


__all__ = [
    "ToolCallRequest",
    "ToolCallResult",
    "ToolCallDecision",
    "ResultDecision",
    "InferenceDecision",
    "HistoryItem",
    "History",
    "FunctionInfo",
]
