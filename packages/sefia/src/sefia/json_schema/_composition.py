from copy import deepcopy

from typing_extensions import final

from ._document import SchemaNode
from ._json import JsonObject, JsonValue


@final
class DefinitionRegistry:
    def __init__(self, definitions: JsonObject):
        self._definitions = definitions

    def import_schema(self, schema: JsonObject, *, namespace: str) -> JsonObject:
        imported = deepcopy(schema)
        local = SchemaNode(imported).take_definitions()
        names = {name: f"{namespace}__{name}" for name in local}
        _rewrite_references(imported, names)

        for name, definition in local.items():
            target = names[name]
            if target in self._definitions:
                raise ValueError(f"duplicate JSON Schema namespace: {namespace!r}")
            rewritten = deepcopy(definition)
            _rewrite_references(rewritten, names)
            self._definitions[target] = rewritten
        return imported


def _rewrite_references(node: JsonValue, names: dict[str, str]) -> None:
    if not isinstance(node, dict):
        return
    for cursor in SchemaNode(node).walk():
        raw_reference = cursor.node.reference
        if raw_reference is None:
            continue
        reference = cursor.node.local_reference
        if reference is None:
            raise ValueError(f"unsupported JSON Schema reference: {raw_reference!r}")
        if reference.definition not in names:
            raise ValueError(
                f"unresolved local JSON Schema reference: {raw_reference!r}"
            )
        cursor.node.set_local_reference(
            reference.with_definition(names[reference.definition])
        )
