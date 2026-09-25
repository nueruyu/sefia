from typing import TypeAlias

from sefia.llm.json import JsonCompatible

JsonObject: TypeAlias = dict[str, JsonCompatible]
