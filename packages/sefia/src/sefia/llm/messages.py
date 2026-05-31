from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a single message in a conversation with an LLM."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[Any] | None = None
    tool_call_id: str | None = None  # Required for role="tool" messages
    tool_calls: list[Any] | None = (
        None  # Present on role="assistant" messages with tool calls
    )


class ToolCall(BaseModel):
    """Represents a tool call requested by the LLM."""

    id: str
    function: dict[str, Any]  # {"name": "...", "arguments": "..."}


class LLMResponse(BaseModel):
    """Represents a response from an LLM."""

    model: str | None = None
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
    stop_reason: str | None = None
    cost: float | None = None
