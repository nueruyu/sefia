from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field


class ToolConflictError(Exception):
    """Raised when two tools with the same name are found."""

    pass


@dataclass(frozen=True)
class Tool:
    """Represents a callable tool with its schema."""

    function: Callable[..., Any]
    schema: dict[str, Any]


class ToolRegistry:
    """
    Stores and provides access to registered tools.
    """

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def add(self, func: Callable[..., Any], schema: dict) -> None:
        """Adds a tool function and its schema to the registry."""
        tool_name = schema["function"]["name"]
        if tool_name in self.tools:
            raise ToolConflictError(
                f"A tool with the name '{tool_name}' already exists."
            )

        self.tools[tool_name] = Tool(function=func, schema=schema)

    def get(self, name: str) -> Tool | None:
        """Gets a tool by its name."""
        return self.tools.get(name)

    def get_all_schemas(self) -> list[dict]:
        """Returns the JSON schemas for all registered tools."""
        return [tool.schema for tool in self.tools.values()]


class LLMToolCall(BaseModel):
    """A tool call requested by the LLM, before an ID is assigned."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    """Represents a request to call a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolCallResult(BaseModel):
    """Represents the result of a single tool call."""

    tool_call_id: str
    result: Any


class ToolCallDecision(BaseModel):
    """A decision to call one or more tools."""

    calls: list[ToolCallRequest]


class FinalAnswerDecision(BaseModel):
    """A decision to return the final answer."""

    answer: Any


InferenceDecision = ToolCallDecision | FinalAnswerDecision
HistoryItem = ToolCallDecision | ToolCallResult


class InferenceHistory(BaseModel):
    """Represents the persisted history of an inference process."""

    items: list[HistoryItem] = Field(default_factory=list)
