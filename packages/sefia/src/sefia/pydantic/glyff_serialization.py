from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Callable

from glyff.interfaces import ArgsHasher, Serializer
from glyff.serialization.helpers import (
    build_hashable_args,
    default_to_hashable,
)
from pydantic import TypeAdapter

from .json_utils import pydantic_json_default


def _json_stable_dumps(data: Any) -> str:
    return json.dumps(
        data, sort_keys=True, default=pydantic_json_default, separators=(",", ":")
    )


class SefiaSerializer(Serializer):
    """
    A Serializer implementation for sefia that preserves pydantic/dataclass
    compatibility while serializing to stable JSON bytes.
    """

    async def serialize(self, value: Any, type_hint: type) -> bytes:
        adapter = TypeAdapter(type_hint)
        json_compatible = adapter.dump_python(value, mode="json")
        return _json_stable_dumps(json_compatible).encode("utf-8")

    async def deserialize(self, data: bytes, type_hint: type) -> Any:
        adapter = TypeAdapter(type_hint)
        return adapter.validate_json(data)


class SefiaArgsHasher(ArgsHasher):
    """
    Deterministic args hasher aligned with glyff's build_hashable_args and
    sefia's JSON conversion rules.
    """

    def hash_args(
        self, func: Callable, sig: inspect.Signature, args: tuple, kwargs: dict
    ) -> str:
        args_dict = build_hashable_args(func, sig, args, kwargs, default_to_hashable)
        stable_repr = _json_stable_dumps(args_dict)
        hasher = hashlib.sha256()
        hasher.update(stable_repr.encode("utf-8"))
        return hasher.hexdigest()
