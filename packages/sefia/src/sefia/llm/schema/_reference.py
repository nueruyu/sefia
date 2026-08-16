from dataclasses import dataclass

from typing_extensions import final

from ._json_pointer import decode_token, encode_token

_DEFINITION_PREFIXES = ("#/$defs/", "#/definitions/")


@final
@dataclass(frozen=True)
class LocalDefinitionRef:
    name: str

    @classmethod
    def parse(cls, value: object) -> "LocalDefinitionRef | None":
        if not isinstance(value, str):
            return None
        for prefix in _DEFINITION_PREFIXES:
            if not value.startswith(prefix):
                continue
            token = value.removeprefix(prefix)
            if "/" in token:
                return None
            name = decode_token(token)
            return cls(name) if name is not None else None
        return None

    def render(self) -> str:
        return f"#/$defs/{encode_token(self.name)}"
