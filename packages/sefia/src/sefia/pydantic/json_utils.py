from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, TypeAdapter


def core_json_default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    try:
        return json.JSONEncoder().default(obj)
    except TypeError as e:
        raise TypeError(
            f"Object of type {type(obj).__name__} is not JSON serializable"
        ) from e


def pydantic_json_default(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return TypeAdapter(type(obj)).dump_python(obj, mode="json")
    return core_json_default(obj)
