import inspect
from dataclasses import dataclass
from typing import Any, Callable

# Only ``self``/``cls`` carry tools — by convention, with no marker. Tool
# dependencies are expressed through classes; plain-function parameters are
# always task data.
_RECEIVER_NAMES = ("self", "cls")


@dataclass(frozen=True)
class Capability:
    """An ``@infer`` call's receiver and its declared type.

    ``declared`` is the receiver's annotation when present (a surface
    ``Protocol`` selecting this method's tools), else ``None``.
    """

    value: object
    declared: Any


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
    def capabilities(self) -> list[Capability]:
        """The call's capability parameters: its receiver plus surface type."""
        return [
            Capability(value=value, declared=self.type_hints.get(name))
            for name, value in self.bound_arguments.items()
            if name in _RECEIVER_NAMES
        ]

    @property
    def prompt_arguments(self) -> dict[str, Any]:
        """The task-data arguments — everything except the receiver."""
        return {
            name: value
            for name, value in self.bound_arguments.items()
            if name not in _RECEIVER_NAMES
        }


__all__ = [
    "Capability",
    "ToolCallRequest",
    "ToolCallResult",
    "ToolCallDecision",
    "ResultDecision",
    "InferenceDecision",
    "HistoryItem",
    "FunctionInfo",
]
