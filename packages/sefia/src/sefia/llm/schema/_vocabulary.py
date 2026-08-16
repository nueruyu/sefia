from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import final


class SchemaKeyword(StrEnum):
    ADDITIONAL_PROPERTIES = "additionalProperties"
    ALL_OF = "allOf"
    ANY_OF = "anyOf"
    CONST = "const"
    CONTAINS = "contains"
    CONTENT_SCHEMA = "contentSchema"
    DEPENDENT_SCHEMAS = "dependentSchemas"
    DEPENDENT_REQUIRED = "dependentRequired"
    DEFINITIONS = "$defs"
    DESCRIPTION = "description"
    ELSE = "else"
    IF = "if"
    ITEMS = "items"
    LEGACY_DEFINITIONS = "definitions"
    MAX_ITEMS = "maxItems"
    MAX_PROPERTIES = "maxProperties"
    MIN_ITEMS = "minItems"
    MIN_PROPERTIES = "minProperties"
    NOT = "not"
    ONE_OF = "oneOf"
    PROPERTIES = "properties"
    PATTERN_PROPERTIES = "patternProperties"
    PREFIX_ITEMS = "prefixItems"
    PROPERTY_NAMES = "propertyNames"
    REFERENCE = "$ref"
    REQUIRED = "required"
    TITLE = "title"
    THEN = "then"
    TYPE = "type"
    UNEVALUATED_ITEMS = "unevaluatedItems"
    UNEVALUATED_PROPERTIES = "unevaluatedProperties"


@final
@dataclass(frozen=True)
class LocalDefinitionRef:
    name: str

    @classmethod
    def parse(cls, value: object) -> "LocalDefinitionRef | None":
        if not isinstance(value, str):
            return None
        for prefix in ("#/$defs/", "#/definitions/"):
            if value.startswith(prefix):
                token = value.removeprefix(prefix)
                if "/" in token:
                    return None
                name = _decode_pointer_token(token)
                return cls(name) if name is not None else None
        return None

    def render(self) -> str:
        name = self.name.replace("~", "~0").replace("/", "~1")
        return f"#/$defs/{name}"


def _decode_pointer_token(token: str) -> str | None:
    result: list[str] = []
    index = 0
    while index < len(token):
        if token[index] != "~":
            result.append(token[index])
            index += 1
            continue
        if index + 1 == len(token) or token[index + 1] not in {"0", "1"}:
            return None
        result.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(result)
