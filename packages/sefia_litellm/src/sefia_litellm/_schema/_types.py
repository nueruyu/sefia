from typing import TypeAlias

from sefia.llm.json import JsonCompatible

SchemaObject: TypeAlias = dict[str, JsonCompatible]
