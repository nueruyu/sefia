from dataclasses import dataclass, field
from typing import Any, Callable


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


@dataclass
class TextBlock:
    """A prompt argument value that should be rendered as a raw text block."""

    value: str


@dataclass
class LLMToolCall:
    """A tool call requested by the LLM, before an ID is assigned."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


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
class FinalAnswerDecision:
    """A decision to return the final answer."""

    answer: Any


InferenceDecision = ToolCallDecision | FinalAnswerDecision
HistoryItem = ToolCallDecision | ToolCallResult


@dataclass
class InferenceHistory:
    """Represents the persisted history of an inference process."""

    items: list[HistoryItem] = field(default_factory=list)
