from typing import Protocol

from typing_extensions import final

from sefia.llm.json_schema import (
    JsonObject,
    JsonScalar,
    SchemaNode,
    SchemaPath,
    SchemaKeyword,
)
from sefia.llm.structured_value import StructuredValue

from ._normalization import SchemaEncodingPlan
from ._traversal import matches, resolve

K = SchemaKeyword


class Decoder(Protocol):
    def decode(self, data: StructuredValue) -> StructuredValue: ...


@final
class _IdentityDecoder:
    def decode(self, data: StructuredValue) -> StructuredValue:
        return data


@final
class _DeferredDecoder:
    def __init__(self) -> None:
        self.target: Decoder | None = None

    def decode(self, data: StructuredValue) -> StructuredValue:
        return data if self.target is None else self.target.decode(data)


@final
class _ObjectDecoder:
    def __init__(self, properties: dict[str, Decoder]):
        self._properties = properties

    def decode(self, data: StructuredValue) -> StructuredValue:
        try:
            fields = data.as_object()
        except ValueError:
            return data
        return StructuredValue.object(
            {
                key: self._properties[key].decode(value)
                if isinstance(key, str) and key in self._properties
                else value
                for key, value in fields.items()
            }
        )


@final
class _ArrayDecoder:
    def __init__(self, item: Decoder):
        self._item = item

    def decode(self, data: StructuredValue) -> StructuredValue:
        try:
            values = data.as_array()
        except ValueError:
            return data
        return StructuredValue.array(self._item.decode(item) for item in values)


@final
class _MappingDecoder:
    def __init__(self, key: Decoder, value: Decoder):
        self._key = key
        self._value = value

    def decode(self, data: StructuredValue) -> StructuredValue:
        try:
            entries = data.as_array()
        except ValueError:
            return data
        result: dict[JsonScalar, StructuredValue] = {}
        for entry in entries:
            fields = entry.as_record("mapping entry")
            if set(fields) != {"key", "value"}:
                raise ValueError("mapping entries must contain only key and value")
            key = self._key.decode(fields["key"]).as_scalar("mapping key")
            if key in result:
                raise ValueError(f"duplicate mapping key: {key!r}")
            result[key] = self._value.decode(fields["value"])
        return StructuredValue.object(result)


@final
class _UnionDecoder:
    def __init__(
        self, choices: list[tuple[JsonObject, Decoder]], root: JsonObject
    ) -> None:
        self._choices = choices
        self._root = root

    def decode(self, data: StructuredValue) -> StructuredValue:
        for schema, decoder in self._choices:
            if matches(data.to_python(), schema, self._root):
                return decoder.decode(data)
        return data


@final
class DecoderFactory:
    def __init__(self, root: JsonObject, plan: SchemaEncodingPlan):
        self._root = root
        self._mapping_paths = plan.mapping_paths
        self._cache: dict[SchemaPath, _DeferredDecoder] = {}

    def build(self, schema: JsonObject, path: SchemaPath = ()) -> Decoder:
        schema, path = resolve(schema, self._root, path)
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        deferred = _DeferredDecoder()
        self._cache[path] = deferred
        deferred.target = self._build_node(schema, path)
        return deferred

    def _build_node(self, schema: JsonObject, path: SchemaPath) -> Decoder:
        node = SchemaNode(schema)
        if path in self._mapping_paths:
            items = node.items()
            properties = items.properties() if items is not None else {}
            if set(properties) != {"key", "value"}:
                raise ValueError("lowered mapping schema is missing key/value entries")
            return _MappingDecoder(
                self.build(
                    properties["key"].value,
                    (*path, K.ITEMS, K.PROPERTIES, "key"),
                ),
                self.build(
                    properties["value"].value,
                    (*path, K.ITEMS, K.PROPERTIES, "value"),
                ),
            )
        alternatives = node.any_of()
        if alternatives:
            return _UnionDecoder(
                [
                    (
                        alternative.value,
                        self.build(alternative.value, (*path, K.ANY_OF, index)),
                    )
                    for index, alternative in enumerate(alternatives)
                ],
                self._root,
            )
        if node.type == "object":
            properties = node.properties()
            return _ObjectDecoder(
                {
                    name: self.build(child.value, (*path, K.PROPERTIES, name))
                    for name, child in properties.items()
                }
            )
        if node.type == "array":
            items = node.items()
            if items is not None:
                return _ArrayDecoder(self.build(items.value, (*path, K.ITEMS)))
        return _IdentityDecoder()
