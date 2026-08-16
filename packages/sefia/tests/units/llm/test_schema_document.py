import pytest

from sefia.llm.schema import JsonSchemaDocument


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
