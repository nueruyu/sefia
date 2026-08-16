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
