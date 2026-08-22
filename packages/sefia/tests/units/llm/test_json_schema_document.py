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

    assert imported == {"$ref": "#/$defs/search__Item"}
    assert definitions == {"search__Item": {"type": "string"}}
    assert fragment == {
        "$ref": "#/$defs/Item",
        "$defs": {"Item": {"type": "string"}},
    }


def test_definition_registry_keeps_fragment_reference_graphs_separate() -> None:
    definitions: JsonObject = {}
    registry = DefinitionRegistry(definitions)
    first: JsonObject = {
        "$ref": "#/$defs/A",
        "$defs": {
            "A": {"$ref": "#/$defs/B"},
            "B": {"$ref": "#/$defs/C"},
            "C": {"type": "string"},
        },
    }
    second: JsonObject = {
        "$ref": "#/$defs/A",
        "$defs": {
            "A": {"$ref": "#/$defs/B"},
            "B": {"$ref": "#/$defs/C"},
            "C": {"type": "integer"},
        },
    }

    first_import = registry.import_schema(first, namespace="fragment_0")
    second_import = registry.import_schema(second, namespace="fragment_1")

    assert first_import == {"$ref": "#/$defs/fragment_0__A"}
    assert second_import == {"$ref": "#/$defs/fragment_1__A"}
    assert definitions == {
        "fragment_0__A": {"$ref": "#/$defs/fragment_0__B"},
        "fragment_0__B": {"$ref": "#/$defs/fragment_0__C"},
        "fragment_0__C": {"type": "string"},
        "fragment_1__A": {"$ref": "#/$defs/fragment_1__B"},
        "fragment_1__B": {"$ref": "#/$defs/fragment_1__C"},
        "fragment_1__C": {"type": "integer"},
    }


def test_definition_registry_does_not_deduplicate_identical_definitions() -> None:
    definitions: JsonObject = {}
    registry = DefinitionRegistry(definitions)
    fragment: JsonObject = {
        "$ref": "#/$defs/Item",
        "$defs": {"Item": {"type": "string"}},
    }

    registry.import_schema(fragment, namespace="fragment_0")
    registry.import_schema(fragment, namespace="fragment_1")

    assert definitions == {
        "fragment_0__Item": {"type": "string"},
        "fragment_1__Item": {"type": "string"},
    }


def test_definition_registry_preserves_escaped_reference_paths() -> None:
    definitions: JsonObject = {}
    registry = DefinitionRegistry(definitions)
    fragment: JsonObject = {
        "$ref": "#/$defs/A~1B~0C/properties/x~1y",
        "$defs": {
            "A/B~C": {
                "type": "object",
                "properties": {"x/y": {"type": "string"}},
            }
        },
    }

    imported = registry.import_schema(fragment, namespace="fragment")

    assert imported == {"$ref": "#/$defs/fragment__A~1B~0C/properties/x~1y"}
    assert definitions == {
        "fragment__A/B~C": {
            "type": "object",
            "properties": {"x/y": {"type": "string"}},
        }
    }


def test_definition_registry_imports_legacy_definitions() -> None:
    definitions: JsonObject = {}
    registry = DefinitionRegistry(definitions)
    fragment: JsonObject = {
        "$ref": "#/definitions/Item",
        "definitions": {"Item": {"type": "string"}},
    }

    imported = registry.import_schema(fragment, namespace="legacy")

    assert imported == {"$ref": "#/$defs/legacy__Item"}
    assert definitions == {"legacy__Item": {"type": "string"}}


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("#/$defs/Missing", "unresolved local JSON Schema reference"),
        ("other.json#/$defs/Item", "unsupported JSON Schema reference"),
    ],
)
def test_definition_registry_rejects_unbundleable_references(
    reference: str, message: str
) -> None:
    registry = DefinitionRegistry({})
    fragment: JsonObject = {"$ref": reference}

    with pytest.raises(ValueError, match=message):
        registry.import_schema(fragment, namespace="fragment")
