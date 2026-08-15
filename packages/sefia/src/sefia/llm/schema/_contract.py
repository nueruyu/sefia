from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from typing_extensions import final, override

SchemaPath = tuple[str | int, ...]


@final
@dataclass(frozen=True)
class LLMSchema:
    """A logical structured-output schema produced by Sefia."""

    schema: dict[str, Any]
    raw_schema_paths: frozenset[SchemaPath] = frozenset()


class PreparedLLMSchema(ABC):
    """A schema adapted for an LLM client and its inverse transformations."""

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]: ...

    @abstractmethod
    def decode(self, data: Any) -> Any: ...

    def normalize_stream_path(self, path: SchemaPath) -> SchemaPath | None:
        return path


@final
class IdentityPreparedLLMSchema(PreparedLLMSchema):
    def __init__(self, schema: LLMSchema):
        self._schema = deepcopy(schema.schema)

    @property
    @override
    def schema(self) -> dict[str, Any]:
        return deepcopy(self._schema)

    @override
    def decode(self, data: Any) -> Any:
        return data
