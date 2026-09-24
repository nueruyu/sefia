from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from .structured_data import StructuredData


@dataclass(frozen=True, init=False)
class Message:
    """An immutable provider-neutral message sent to an LLM."""

    role: Literal["system", "developer", "user", "assistant", "tool"]
    _content: str | list[Any] | None = field(repr=False)
    tool_call_id: str | None
    tool_calls: tuple[ToolCall, ...] | None

    def __init__(
        self,
        role: Literal["system", "developer", "user", "assistant", "tool"],
        content: str | list[Any] | None = None,
        tool_call_id: str | None = None,
        tool_calls: Sequence[ToolCall] | None = None,
    ) -> None:
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "_content", _copy_content(content))
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(
            self,
            "tool_calls",
            None if tool_calls is None else tuple(tool_calls),
        )

    @property
    def content(self) -> str | list[Any] | None:
        return _copy_content(self._content)


@dataclass(frozen=True)
class ToolCall:
    """An immutable provider-neutral tool call requested by the LLM."""

    id: str
    name: str
    arguments: StructuredData


def _copy_content(value: Any) -> Any:
    if isinstance(value, list):
        return [_copy_content(item) for item in cast(list[Any], value)]
    if isinstance(value, dict):
        return {
            key: _copy_content(item)
            for key, item in cast(dict[Any, Any], value).items()
        }
    if isinstance(value, tuple):
        return tuple(_copy_content(item) for item in cast(tuple[Any, ...], value))
    return value


@dataclass
class LLMCompletion:
    """A provider-neutral completion returned by an ``LLMClient``."""

    model: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])
    usage: dict[str, Any] | None = None
    stop_reason: str | None = None
    cost: float | None = None
    structured_output: StructuredData | None = None
