import pytest

from sefia.llm.schema import JsonSchemaDocument, LocalDefinitionRef, SchemaNode


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


def test_schema_node_owns_common_rewrites() -> None:
    union = SchemaNode({"oneOf": [{"type": "string"}]})
    union.normalize_one_of()
    assert union.one_of() == []
    assert len(union.any_of()) == 1

    mapping = SchemaNode(
        {
            "type": "object",
            "title": "Labels",
            "additionalProperties": {"type": "integer"},
            "minProperties": 1,
        }
    )

    value_schema = mapping.additional_properties()
    assert isinstance(value_schema, SchemaNode)
    mapping.replace_with_mapping_entries({"type": "string"}, value_schema.value)

    assert mapping.type == "array"
    assert mapping.items() is not None
    assert mapping.value["minItems"] == 1
