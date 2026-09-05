"""Stable constructors for test data used across sefia extensions."""

from dataclasses import replace
from typing import Any

from .._history import StepHistory
from .._interfaces.middleware import StepContext
from .._tool_system import ToolRegistry
from ..inference import FunctionInfo, HistoryItem, ToolCallRequest
from ..llm import RejectedDecision
from ..llm.step_decision import DecisionSpec
from ..llm.transports import DecisionRequest


def _test_function() -> str:
    """Execute the test task."""
    return ""


def make_function_info(
    *,
    qualname: str = "test",
    instructions: str = "instructions",
    bound_arguments: dict[str, Any] | None = None,
    type_hints: dict[str, Any] | None = None,
    return_type: Any = str,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> FunctionInfo:
    """Build ordinary function metadata with explicit test overrides."""
    base = FunctionInfo.create(_test_function, (), {})
    return replace(
        base,
        qualname=qualname,
        instructions=instructions,
        bound_arguments={} if bound_arguments is None else bound_arguments,
        type_hints={} if type_hints is None else type_hints,
        return_type=return_type,
        args=args,
        kwargs={} if kwargs is None else kwargs,
    )


def make_decision_request(
    decision_spec: DecisionSpec,
    *,
    function: FunctionInfo | None = None,
    history: tuple[HistoryItem, ...] = (),
    rejected: RejectedDecision | None = None,
) -> DecisionRequest:
    """Build a decision request with ordinary function metadata."""
    return DecisionRequest(
        function=make_function_info() if function is None else function,
        decision_spec=decision_spec,
        history=history,
        rejected=rejected,
    )


def make_tool_call_request(
    *,
    id: str = "call-1",
    name: str = "tool",
    arguments: dict[str, Any] | None = None,
) -> ToolCallRequest:
    """Build a tool-call request with stable defaults for consumer tests."""
    return ToolCallRequest(
        id=id,
        name=name,
        arguments={} if arguments is None else arguments,
    )


def make_step_context(
    *,
    step: int = 0,
    history: StepHistory | None = None,
    tool_registry: ToolRegistry | None = None,
) -> StepContext:
    """Build an isolated middleware context for one inference step."""
    return StepContext(
        step=step,
        history=StepHistory() if history is None else history,
        tool_registry=ToolRegistry() if tool_registry is None else tool_registry,
    )


__all__ = [
    "make_decision_request",
    "make_function_info",
    "make_step_context",
    "make_tool_call_request",
]
