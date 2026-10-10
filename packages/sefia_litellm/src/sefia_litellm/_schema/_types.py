from typing import TypeAlias

from sefia.llm import JsonCompatible

SchemaObject: TypeAlias = dict[str, JsonCompatible]
