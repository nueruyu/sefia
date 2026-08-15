from typing import Any, Protocol, cast

from typing_extensions import final

from ._traversal import matches, resolve


class Decoder(Protocol):
    def decode(self, data: Any) -> Any: ...


@final
class _IdentityDecoder:
    def decode(self, data: Any) -> Any:
        return data


@final
class _DeferredDecoder:
    def __init__(self) -> None:
        self.target: Decoder | None = None

    def decode(self, data: Any) -> Any:
        return data if self.target is None else self.target.decode(data)


@final
class _ObjectDecoder:
    def __init__(self, properties: dict[str, Decoder]):
        self._properties = properties

    def decode(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return {
            key: self._properties[key].decode(value)
            if key in self._properties
            else value
            for key, value in cast(dict[str, Any], data).items()
        }


@final
class _ArrayDecoder:
    def __init__(self, item: Decoder):
        self._item = item

    def decode(self, data: Any) -> Any:
        if not isinstance(data, list):
            return data
        return [self._item.decode(item) for item in cast(list[Any], data)]


@final
class _MappingDecoder:
    def __init__(self, key: Decoder, value: Decoder):
        self._key = key
        self._value = value

    def decode(self, data: Any) -> Any:
        if not isinstance(data, list):
            return data
        result: dict[Any, Any] = {}
        for entry in cast(list[Any], data):
            if not isinstance(entry, dict):
                raise ValueError("mapping entries must be objects")
            entry_map = cast(dict[str, Any], entry)
            if set(entry_map) != {"key", "value"}:
                raise ValueError("mapping entries must contain only key and value")
            key = self._key.decode(entry_map["key"])
            try:
                if key in result:
                    raise ValueError(f"duplicate mapping key: {key!r}")
                result[key] = self._value.decode(entry_map["value"])
            except TypeError as error:
                raise ValueError("mapping keys must be hashable") from error
        return result


@final
class _UnionDecoder:
    def __init__(
        self, choices: list[tuple[dict[str, Any], Decoder]], root: dict[str, Any]
    ):
        self._choices = choices
        self._root = root

    def decode(self, data: Any) -> Any:
        for schema, decoder in self._choices:
            if matches(data, schema, self._root):
                return decoder.decode(data)
        return data


@final
class DecoderFactory:
    def __init__(self, root: dict[str, Any], mapping_ids: set[int]):
        self._root = root
        self._mapping_ids = mapping_ids
        self._cache: dict[int, _DeferredDecoder] = {}

    def build(self, schema: dict[str, Any]) -> Decoder:
        schema = resolve(schema, self._root)
        cached = self._cache.get(id(schema))
        if cached is not None:
            return cached
        deferred = _DeferredDecoder()
        self._cache[id(schema)] = deferred
        deferred.target = self._build_node(schema)
        return deferred

    def _build_node(self, schema: dict[str, Any]) -> Decoder:
        if id(schema) in self._mapping_ids:
            properties = cast(
                dict[str, Any], cast(dict[str, Any], schema["items"])["properties"]
            )
            return _MappingDecoder(
                self.build(cast(dict[str, Any], properties["key"])),
                self.build(cast(dict[str, Any], properties["value"])),
            )
        any_of = schema.get("anyOf")
        if isinstance(any_of, list):
            choices = [
                (candidate, self.build(candidate))
                for item in cast(list[Any], any_of)
                if isinstance(item, dict)
                for candidate in [cast(dict[str, Any], item)]
            ]
            return _UnionDecoder(choices, self._root)
        if schema.get("type") == "object":
            properties = schema.get("properties")
            if isinstance(properties, dict):
                return _ObjectDecoder(
                    {
                        name: self.build(cast(dict[str, Any], child))
                        for name, child in cast(dict[str, Any], properties).items()
                        if isinstance(child, dict)
                    }
                )
        if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
            return _ArrayDecoder(self.build(cast(dict[str, Any], schema["items"])))
        return _IdentityDecoder()
