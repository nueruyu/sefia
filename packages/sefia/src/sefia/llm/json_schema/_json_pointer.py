from ._json import JsonValue


def encode_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def decode_token(value: str) -> str | None:
    result: list[str] = []
    index = 0
    while index < len(value):
        if value[index] != "~":
            result.append(value[index])
            index += 1
            continue
        if index + 1 == len(value) or value[index + 1] not in {"0", "1"}:
            return None
        result.append("~" if value[index + 1] == "0" else "/")
        index += 2
    return "".join(result)


def resolve_tokens(value: JsonValue, tokens: tuple[str, ...]) -> JsonValue | None:
    current = value
    for token in tokens:
        if isinstance(current, dict):
            if token not in current:
                return None
            current = current[token]
            continue
        if isinstance(current, list) and token.isdigit():
            index = int(token)
            if index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current
