from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import final

from ._document import SchemaNode
from ._json import JsonObject, JsonValue


@final
@dataclass(frozen=True)
class ImportedSchema:
    schema: JsonObject
    definitions: JsonObject


@final
class DefinitionRegistry:
    def __init__(self, definitions: JsonObject):
        self._definitions = definitions
        self._reserved_names: set[str] = set()

    def reserve(self, names: Iterable[str]) -> None:
        self._reserved_names.update(names)

    def import_schema(self, schema: JsonObject, *, namespace: str) -> ImportedSchema:
        local = SchemaNode(schema).take_definitions()
        names = {
            name: self._target_name(namespace, name, definition)
            for name, definition in local.items()
        }
        _rewrite_references(schema, names)

        imported: JsonObject = {}
        for name, definition in local.items():
            target = names[name]
            if target in self._definitions:
                continue
            rewritten = deepcopy(definition)
            _rewrite_references(rewritten, names)
            self._definitions[target] = rewritten
            imported[target] = rewritten
        return ImportedSchema(schema, imported)

    def _target_name(self, namespace: str, name: str, definition: JsonValue) -> str:
        if name not in self._definitions:
            return name
        if name not in self._reserved_names and self._definitions[name] == definition:
            return name

        base = f"{namespace}__{name}"
        candidate = base
        suffix = 2
        while candidate in self._definitions:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate


def _rewrite_references(node: JsonValue, names: dict[str, str]) -> None:
    if not isinstance(node, dict):
        return
    for cursor in SchemaNode(node).walk():
        reference = cursor.node.local_reference
        if reference is None or reference.definition not in names:
            continue
        cursor.node.set_local_reference(
            reference.with_definition(names[reference.definition])
        )
