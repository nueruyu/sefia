from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .json import JsonCompatible


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
    return cast("JsonCompatible", value)
