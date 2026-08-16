from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, cast


def _to_serializable(value: Any, exclude_none: bool) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_serializable(asdict(value), exclude_none=exclude_none)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in cast(dict[str, Any], value).items():
            converted = _to_serializable(item, exclude_none=exclude_none)
            if exclude_none and converted is None:
                continue
            result[key] = converted
        return result
    if isinstance(value, list):
        return [
            _to_serializable(item, exclude_none=exclude_none)
            for item in cast(list[Any], value)
        ]
    return value


@dataclass
class Message:
    """Represents a single message in a conversation with an LLM."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[Any] | None = None
    tool_call_id: str | None = None  # Required for role="tool" messages
    tool_calls: list[Any] | None = (
        None  # Present on role="assistant" messages with tool calls
    )

    def to_dict(self, *, exclude_none: bool = False) -> dict[str, Any]:
        return _to_serializable(self, exclude_none=exclude_none)


@dataclass
class ToolCall:
    """Represents a tool call requested by the LLM."""

    id: str
    function: dict[str, Any]  # {"name": "...", "arguments": "..."}


@dataclass
class LLMResponse:
    """Represents a response from an LLM."""

    model: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])
    usage: dict[str, Any] | None = None
    stop_reason: str | None = None
    cost: float | None = None
    structured_output: object | None = None
