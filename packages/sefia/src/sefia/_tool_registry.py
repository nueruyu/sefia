from dataclasses import dataclass
from typing import Any, Callable

from .exceptions import ToolConflictError


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
