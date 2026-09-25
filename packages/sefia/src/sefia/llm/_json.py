import json
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import TypeAlias, cast

from typing_extensions import final

JsonCompatible: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonCompatible"]
    | dict[str, "JsonCompatible"]
)


JsonPath: TypeAlias = tuple[str | int, ...]


class JsonMaterializer(ABC):
    """Materialize application values as detached JSON-compatible structures.

    All object keys must be strings and all floats finite. Returned containers
    must be detached from application containers; implementations must not retain
    mutable aliases.
    The caller may immediately transfer ownership to a snapshot.
    """

    @abstractmethod
    def materialize(self, value: object) -> JsonCompatible: ...


@final
class JsonSnapshot:
    """An owned JSON representation with explicit detached projection."""

    __slots__ = ("__tree",)
    __tree: JsonCompatible

    def __init__(self) -> None:
        raise TypeError("Use JsonSnapshot.capture() or a shape constructor")

    @classmethod
    def capture(cls, value: JsonCompatible) -> "JsonSnapshot":
        return cls.__wrap(copy_json(value))

    @classmethod
    def _from_owned(cls, value: JsonCompatible) -> "JsonSnapshot":
        """Adopt a detached graph whose mutable aliases the caller relinquishes."""
        return cls.__wrap(require_json_compatible(value))

    @classmethod
    def __wrap(cls, value: JsonCompatible) -> "JsonSnapshot":
        snapshot = object.__new__(cls)
        snapshot.__tree = value
        return snapshot

    @classmethod
    def parse_json(cls, text: str) -> "JsonSnapshot":
        return cls._from_owned(json.loads(text))

    @classmethod
    def from_scalar(cls, value: str | int | float | bool | None) -> "JsonSnapshot":
        return cls.__wrap(require_json_scalar(value))

    @classmethod
    def from_array(cls, values: Iterable["JsonSnapshot"]) -> "JsonSnapshot":
        return cls.__wrap([value.__tree for value in values])

    @classmethod
    def from_object(cls, fields: Mapping[str, "JsonSnapshot"]) -> "JsonSnapshot":
        if not all(
            isinstance(key, str) for key in cast(Mapping[object, object], fields)
        ):
            raise ValueError("JSON object keys must be strings")
        return cls.__wrap({key: value.__tree for key, value in fields.items()})

    def to_json_compatible(self) -> JsonCompatible:
        return copy_json(self.__tree)

    def as_object(
        self, description: str = "JSON snapshot"
    ) -> dict[str, "JsonSnapshot"]:
        if not isinstance(self.__tree, dict):
            raise ValueError(f"{description} must be an object")
        return {key: self.__wrap(value) for key, value in self.__tree.items()}

    def as_array(self, description: str = "JSON snapshot") -> list["JsonSnapshot"]:
        if not isinstance(self.__tree, list):
            raise ValueError(f"{description} must be an array")
        return [self.__wrap(value) for value in self.__tree]

    def as_string(self, description: str = "JSON snapshot") -> str:
        if not isinstance(self.__tree, str):
            raise ValueError(f"{description} must be a string")
        return self.__tree

    def as_scalar(
        self, description: str = "JSON snapshot"
    ) -> str | int | float | bool | None:
        if self.__tree is None or isinstance(self.__tree, str | int | float | bool):
            return self.__tree
        raise ValueError(f"{description} must be a scalar")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JsonSnapshot):
            return NotImplemented
        return self.__tree == other.__tree

    __hash__ = None  # pyright: ignore[reportAssignmentType]


def require_json_scalar(value: object) -> str | int | float | bool | None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ValueError(f"Unsupported JSON scalar: {value!r}")


def copy_json(value: object) -> JsonCompatible:
    if isinstance(value, list):
        return [copy_json(item) for item in cast(list[object], value)]
    if isinstance(value, dict):
        result: dict[str, JsonCompatible] = {}
        for key, item in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            result[key] = copy_json(item)
        return result
    return require_json_scalar(value)


def require_json_compatible(value: object) -> JsonCompatible:
    """Validate without allocating a replacement tree."""
    if isinstance(value, list):
        for item in cast(list[object], value):
            require_json_compatible(item)
    elif isinstance(value, dict):
        for key, item in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            require_json_compatible(item)
    else:
        return require_json_scalar(value)
    return cast(JsonCompatible, value)
