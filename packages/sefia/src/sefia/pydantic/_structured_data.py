from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import cast
from uuid import UUID

from pydantic import BaseModel, TypeAdapter

from ..llm.json_schema import JsonScalar
from ..llm.structured_data import StructuredData


def to_structured_data(value: object) -> StructuredData:
    if isinstance(value, Enum):
        return to_structured_data(value.value)
    if value is None or isinstance(value, str | int | float | bool):
        return StructuredData.from_scalar(value)
    if isinstance(value, BaseModel):
        serialized = TypeAdapter(type(value)).dump_python(value, mode="json")
        return to_structured_data(serialized)
    if is_dataclass(value) and not isinstance(value, type):
        return StructuredData.from_object(
            {
                field.name: to_structured_data(getattr(value, field.name))
                for field in fields(value)
            }
        )
    if isinstance(value, UUID):
        return StructuredData.from_scalar(str(value))
    if isinstance(value, datetime | date):
        return StructuredData.from_scalar(value.isoformat())
    if isinstance(value, Mapping):
        entries: dict[JsonScalar, StructuredData] = {}
        for key, item in cast(Mapping[object, object], value).items():
            structured_key = _dump_key(key)
            if structured_key in entries:
                raise ValueError(
                    "Mapping contains keys that normalize to the same structured "
                    f"key: {structured_key!r}"
                )
            entries[structured_key] = to_structured_data(item)
        return StructuredData.from_mapping(entries)
    if isinstance(value, list | tuple):
        sequence = cast(list[object] | tuple[object, ...], value)
        return StructuredData.from_array(to_structured_data(item) for item in sequence)
    return StructuredData.from_scalar(str(value))


def _dump_key(value: object) -> JsonScalar:
    structured = to_structured_data(value).tree
    if structured is None or isinstance(structured, str | int | float | bool):
        return structured
    return str(structured)


__all__ = ["to_structured_data"]
