import pytest

from sefia.llm.json_schema import (
    DefinitionRegistry,
    JsonObject,
    JsonSchemaDocument,
    LocalDefinitionRef,
)


def test_schema_document_rejects_non_json_values() -> None:
    with pytest.raises(TypeError, match="not a JSON object"):
        JsonSchemaDocument.from_mapping({"type": object()})


def test_schema_document_owns_and_returns_defensive_copies() -> None:
    source = {"type": "object", "properties": {"name": {"type": "string"}}}
    document = JsonSchemaDocument.from_mapping(source)

    source["type"] = "string"
    exported = document.to_dict()
    exported["type"] = "array"

    assert document.root().type == "object"


def test_schema_node_exposes_structure_without_untyped_indexing() -> None:
    document = JsonSchemaDocument.from_mapping(
        {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}
            },
            "$defs": {
                "Item": {
                    "type": "object",
                    "required": ["name"],
                    "additionalProperties": False,
                }
            },
        }
    )

    root = document.root()
    item = root.properties()["items"].child("items")

    assert item is not None
    assert item.reference == "#/$defs/Item"
    definition = root.definitions()["Item"]
    assert definition.strings("required") == ("name",)
    assert definition.additional_properties() is False
    assert {cursor.path for cursor in document.walk()} >= {
        (),
        ("properties", "items"),
        ("properties", "items", "items"),
        ("$defs", "Item"),
    }
    resolved = item.resolve_local_reference(root)
    assert resolved is not None
    assert resolved.value == definition.value


def test_local_definition_reference_handles_json_pointer_escaping() -> None:
    reference = LocalDefinitionRef("group/item~name")

    assert reference.render() == "#/$defs/group~1item~0name"
    assert LocalDefinitionRef.parse(reference.render()) == reference
    assert LocalDefinitionRef.parse("#/definitions/Legacy") == LocalDefinitionRef(
        "Legacy"
    )
    assert LocalDefinitionRef.parse("#/$defs/nested/properties/name") == (
        LocalDefinitionRef("nested", ("properties", "name"))
    )
    assert LocalDefinitionRef.parse("#/$defs/invalid~escape") is None

    nested = LocalDefinitionRef("User", ("properties", "name"))
    assert nested.with_definition("tool__User").render() == (
        "#/$defs/tool__User/properties/name"
    )
    assert nested.resolve_from(
        {"User": {"properties": {"name": {"type": "string"}}}}
    ) == {"type": "string"}


def test_definition_registry_imports_definitions_and_rewrites_references() -> None:
    definitions: JsonObject = {}
    registry = DefinitionRegistry(definitions)
    fragment: JsonObject = {
        "$ref": "#/$defs/Item",
        "$defs": {"Item": {"type": "string"}},
    }

    imported = registry.import_schema(fragment, namespace="search")

    assert imported == {"$ref": "#/$defs/Item"}
    assert definitions == {"Item": {"type": "string"}}


def test_definition_registry_renames_conflicting_definitions() -> None:
    definitions: JsonObject = {"Item": {"type": "string"}}
    registry = DefinitionRegistry(definitions)
    fragment: JsonObject = {
        "$ref": "#/$defs/Item",
        "$defs": {"Item": {"type": "integer"}},
    }

    imported = registry.import_schema(fragment, namespace="search")

    assert imported == {"$ref": "#/$defs/search__Item"}
    assert definitions == {
        "Item": {"type": "string"},
        "search__Item": {"type": "integer"},
    }
