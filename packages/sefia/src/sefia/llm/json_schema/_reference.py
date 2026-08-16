from dataclasses import dataclass

from typing_extensions import final

from ._json import JsonValue
from ._json_pointer import decode_token, encode_token, resolve_tokens

_DEFINITION_PREFIXES = ("#/$defs/", "#/definitions/")


@final
@dataclass(frozen=True)
class LocalDefinitionRef:
    definition: str
    path: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value: str) -> "LocalDefinitionRef | None":
        for prefix in _DEFINITION_PREFIXES:
            if not value.startswith(prefix):
                continue
            encoded_tokens = value.removeprefix(prefix).split("/")
            tokens = tuple(decode_token(token) for token in encoded_tokens)
            if any(token is None for token in tokens):
                return None
            decoded = tuple(token for token in tokens if token is not None)
            return cls(decoded[0], decoded[1:])
        return None

    def render(self) -> str:
        tokens = (self.definition, *self.path)
        return "#/$defs/" + "/".join(encode_token(token) for token in tokens)

    def with_definition(self, definition: str) -> "LocalDefinitionRef":
        return LocalDefinitionRef(definition, self.path)

    def resolve_from(self, definitions: dict[str, JsonValue]) -> JsonValue | None:
        definition = definitions.get(self.definition)
        if definition is None:
            return None
        return resolve_tokens(definition, self.path)
