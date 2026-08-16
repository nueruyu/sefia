from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import final

from ._document import JsonSchemaDocument

from ._path import SchemaPath


@final
@dataclass(frozen=True)
class LLMSchema:
    """A logical structured-output schema produced by Sefia."""

    document: JsonSchemaDocument
    raw_schema_paths: frozenset[SchemaPath] = frozenset()
