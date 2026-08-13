import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from typing_extensions import final, override

from ..streaming import StreamHandler


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolFunctionInspector(ABC):
    @abstractmethod
    def tool_name(self, func: Callable[..., Any]) -> str: ...

    @abstractmethod
    def definition(self, func: Callable[..., Any], *, name: str) -> ToolDefinition: ...

    @abstractmethod
    def bind(
        self, func: Callable[..., Any], arguments: dict[str, Any]
    ) -> dict[str, Any]: ...


class ToolEntry(ABC):
    name: str
    stream_handler: StreamHandler | None
    concurrent: bool = False

    @abstractmethod
    def definition(self) -> ToolDefinition: ...

    @abstractmethod
    async def invoke(self, arguments: dict[str, Any]) -> Any: ...

    @property
    def function(self) -> Callable[..., Any] | None:
        return None


@final
class SignatureToolEntry(ToolEntry):
    def __init__(
        self,
        function: Callable[..., Any],
        *,
        name: str,
        schema_source: Callable[..., Any],
        inspector: ToolFunctionInspector,
        stream_handler: StreamHandler | None = None,
        concurrent: bool = False,
    ):
        self.name = name
        self.stream_handler = stream_handler
        self.concurrent = concurrent
        self._function = function
        self._schema_source = schema_source
        self._inspector = inspector

    @override
    def definition(self) -> ToolDefinition:
        return self._inspector.definition(self._schema_source, name=self.name)

    @override
    async def invoke(self, arguments: dict[str, Any]) -> Any:
        bound = self._inspector.bind(self._function, arguments)
        return await _maybe_await(self._function(**bound))

    @property
    @override
    def function(self) -> Callable[..., Any]:
        return self._function


@final
class JsonSchemaToolEntry(ToolEntry):
    def __init__(
        self,
        handler: Callable[..., Any],
        *,
        name: str,
        parameters: dict[str, Any],
        description: str = "",
        stream_handler: StreamHandler | None = None,
        concurrent: bool = False,
    ):
        self.name = name
        self.stream_handler = stream_handler
        self.concurrent = concurrent
        self._handler = handler
        self._parameters = parameters
        self._description = description

    @override
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self._description,
            parameters=self._parameters,
        )

    @override
    async def invoke(self, arguments: dict[str, Any]) -> Any:
        return await _maybe_await(self._handler(**arguments))

    @property
    @override
    def function(self) -> Callable[..., Any]:
        return self._handler


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
