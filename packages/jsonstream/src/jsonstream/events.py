import dataclasses
from typing import TypeAlias

JsonPath: TypeAlias = tuple[str | int, ...]
JsonScalar: TypeAlias = str | int | float | bool | None


@dataclasses.dataclass(frozen=True)
class StartObject:
    path: JsonPath


@dataclasses.dataclass(frozen=True)
class EndObject:
    path: JsonPath


@dataclasses.dataclass(frozen=True)
class StartArray:
    path: JsonPath


@dataclasses.dataclass(frozen=True)
class EndArray:
    path: JsonPath


@dataclasses.dataclass(frozen=True)
class Key:
    path: JsonPath
    value: str


@dataclasses.dataclass(frozen=True)
class StartString:
    path: JsonPath


@dataclasses.dataclass(frozen=True)
class StringDelta:
    path: JsonPath
    text: str


@dataclasses.dataclass(frozen=True)
class EndString:
    path: JsonPath
    value: str


@dataclasses.dataclass(frozen=True)
class Scalar:
    path: JsonPath
    value: JsonScalar


@dataclasses.dataclass(frozen=True)
class JsonParseError:
    message: str
    fatal: bool


Event: TypeAlias = (
    StartObject
    | EndObject
    | StartArray
    | EndArray
    | Key
    | StartString
    | StringDelta
    | EndString
    | Scalar
    | JsonParseError
)
