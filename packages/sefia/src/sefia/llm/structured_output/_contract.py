from dataclasses import dataclass

from typing_extensions import final

from ..json_schema import JsonSchemaDocument, SchemaPath


@final
@dataclass(frozen=True)
class StructuredOutputSchema:
    document: JsonSchemaDocument
    preserved_schema_paths: frozenset[SchemaPath] = frozenset()
