from typing import Annotated, Any, Callable, TypeVar

from typing_extensions import TypeAliasType

from .._introspection import unwrap_annotation
from ..streaming import StreamHandler


class _RoleMarker:
    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return f"sefia.{self._name}"


_TOOLS = _RoleMarker("Tools")
T = TypeVar("T")

Tools = TypeAliasType("Tools", Annotated[T, _TOOLS], type_params=(T,))
"""A field whose members the model may call as tools."""


def bears_tools(annotation: Any) -> bool:
    metadata, _ = unwrap_annotation(annotation)
    return any(item is _TOOLS for item in metadata)


def role_interface(annotation: Any) -> Any:
    return unwrap_annotation(annotation)[1]


_STREAM_HANDLER_ATTR = "__sefia_stream_handler__"
_CONCURRENT_ATTR = "__sefia_concurrent__"


def set_stream_handler(func: Callable[..., Any], handler: StreamHandler) -> None:
    setattr(func, _STREAM_HANDLER_ATTR, handler)


def get_stream_handler(func: Callable[..., Any]) -> StreamHandler | None:
    return getattr(func, _STREAM_HANDLER_ATTR, None)


def set_concurrent(func: Callable[..., Any]) -> None:
    setattr(func, _CONCURRENT_ATTR, True)


def is_concurrent(func: Callable[..., Any]) -> bool:
    return getattr(func, _CONCURRENT_ATTR, False)
