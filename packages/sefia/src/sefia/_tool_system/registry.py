import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable

from ..exceptions import ToolConflictError
from ..inference import Capability
from ..streaming import StreamHandler
from .entries import (
    JsonSchemaToolEntry,
    SignatureToolEntry,
    ToolEntry,
    ToolFunctionInspector,
)


def _callable_identity(func: Callable[..., Any]) -> Callable[..., Any]:
    return inspect.unwrap(getattr(func, "__func__", func))


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}

    def register(self, tool: ToolEntry) -> None:
        if tool.name in self._tools:
            raise ToolConflictError(
                f"A tool with the name '{tool.name}' already exists."
            )
        self._tools[tool.name] = tool

    def add(
        self,
        func: Callable[..., Any],
        *,
        name: str,
        schema_source: Callable[..., Any] | None = None,
        inspector: ToolFunctionInspector | None = None,
        stream_handler: StreamHandler | None = None,
        concurrent: bool = False,
    ) -> None:
        if inspector is None:
            from ..pydantic._model_backend import PydanticModelBackend

            inspector = PydanticModelBackend()
        self.register(
            SignatureToolEntry(
                func,
                name=name,
                schema_source=schema_source or func,
                inspector=inspector,
                stream_handler=stream_handler,
                concurrent=concurrent,
            )
        )

    def add_json_tool(
        self,
        handler: Callable[..., Any],
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        stream_handler: StreamHandler | None = None,
        concurrent: bool = False,
    ) -> None:
        self.register(
            JsonSchemaToolEntry(
                handler,
                name=name,
                parameters=parameters,
                description=description,
                stream_handler=stream_handler,
                concurrent=concurrent,
            )
        )

    def get(self, name: str) -> ToolEntry | None:
        return self._tools.get(name)

    def get_by_function(self, func: Callable[..., Any]) -> list[ToolEntry]:
        target = _callable_identity(func)
        return [
            tool
            for tool in self._tools.values()
            if tool.function is not None and _callable_identity(tool.function) is target
        ]

    def get_all(self) -> list[ToolEntry]:
        return list(self._tools.values())

    def get_names(self) -> list[str]:
        return list(self._tools.keys())


class ToolCollector(ABC):
    @abstractmethod
    def collect(self, capabilities: list[Capability]) -> ToolRegistry: ...
