from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from typing_extensions import final

SchemaPath = tuple[str | int, ...]


@final
@dataclass(frozen=True)
class LLMSchema:
    """A logical structured-output schema produced by Sefia."""

    schema: dict[str, Any]
    raw_schema_paths: frozenset[SchemaPath] = frozenset()
