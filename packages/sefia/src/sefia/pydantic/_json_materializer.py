from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import cast
from uuid import UUID

from pydantic import BaseModel, TypeAdapter
from typing_extensions import final, override

from ..llm._json_tree import require_json_compatible, require_json_scalar
from ..llm.json import JsonCompatible, JsonMaterializer


@final
class PydanticJsonMaterializer(JsonMaterializer):
    """Materialize Python values as JSON with Pydantic support."""

    @override
    def materialize(self, value: object) -> JsonCompatible:
        return _materialize(value)


def _materialize(value: object) -> JsonCompatible:
    if isinstance(value, Enum):
        return _materialize(value.value)
    if value is None or isinstance(value, str | int | float | bool):
        return require_json_scalar(value)
    if isinstance(value, BaseModel):
        _check_mapping_keys(value)
        return require_json_compatible(
            TypeAdapter(type(value)).dump_python(value, mode="json")
        )
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _materialize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, JsonCompatible] = {}
        for key, item in cast(Mapping[object, object], value).items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            result[key] = _materialize(item)
        return result
    if isinstance(value, list | tuple):
        return [
            _materialize(item)
            for item in cast(list[object] | tuple[object, ...], value)
        ]
    return str(value)


def _check_mapping_keys(value: object) -> None:
    # JSON-mode model dumps stringify mapping keys before normalization sees them.
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            _check_mapping_keys(getattr(value, name))
        if value.model_extra is not None:
            _check_mapping_keys(value.model_extra)
    elif isinstance(value, Mapping):
        for key, item in cast(Mapping[object, object], value).items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            _check_mapping_keys(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _check_mapping_keys(getattr(value, field.name))
    elif isinstance(value, list | tuple):
        for item in cast(list[object] | tuple[object, ...], value):
            _check_mapping_keys(item)
    elif isinstance(value, Enum):
        _check_mapping_keys(value.value)


__all__ = ["PydanticJsonMaterializer"]
