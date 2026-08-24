from ._json import JsonValue


def without_titles(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: without_titles(item) for key, item in value.items() if key != "title"
        }
    if isinstance(value, list):
        return [without_titles(item) for item in value]
    return value


__all__ = ["without_titles"]
