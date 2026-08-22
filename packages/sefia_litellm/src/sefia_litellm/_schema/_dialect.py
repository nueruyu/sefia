from typing_extensions import final

from sefia.llm.json_schema import JsonObject, SchemaKeyword, SchemaNode, SchemaPath

K = SchemaKeyword

_UNSUPPORTED_COMPOSITION = (
    K.ALL_OF,
    K.NOT,
    K.DEPENDENT_REQUIRED,
    K.DEPENDENT_SCHEMAS,
    K.IF,
    K.THEN,
    K.ELSE,
)


@final
class StructuredOutputDialect:
    def adapt(self, schema: JsonObject) -> None:
        for cursor in SchemaNode(schema).walk():
            node = cursor.node
            if node.type == "object":
                self._close_object(node)
            self._replace_one_of(node)

    def validate(self, schema: JsonObject) -> None:
        for cursor in SchemaNode(schema).walk():
            path, node = cursor.path, cursor.node
            if K.ONE_OF in node.value:
                self._unsupported(path, "oneOf is not supported; use a disjoint anyOf")
            for keyword in _UNSUPPORTED_COMPOSITION:
                if keyword in node.value:
                    self._unsupported(path, f"{keyword} is not supported")
            if node.type == "object":
                self._validate_object(path, node)

    @staticmethod
    def _close_object(node: SchemaNode) -> None:
        node.value.setdefault(K.ADDITIONAL_PROPERTIES, False)
        properties = node.object_map(K.PROPERTIES)
        if properties is not None:
            node.value[K.REQUIRED] = list(properties)

    @staticmethod
    def _replace_one_of(node: SchemaNode) -> None:
        alternatives = node.value.pop(K.ONE_OF, None)
        if alternatives is not None:
            node.value[K.ANY_OF] = alternatives

    def _validate_object(self, path: SchemaPath, node: SchemaNode) -> None:
        if node.additional_properties() is not False:
            self._unsupported(
                path, "object schemas must set additionalProperties to false"
            )
        properties = node.properties()
        required_names = set(node.required() or ())
        missing = sorted(set(properties) - required_names)
        if missing:
            self._unsupported(
                path, f"all object properties must be required; missing {missing}"
            )

    @staticmethod
    def _unsupported(path: SchemaPath, detail: str) -> None:
        location = "/".join(map(str, path)) or "<root>"
        raise ValueError(
            "LLM schema is not compatible with strict structured output at "
            f"{location}: {detail}"
        )
