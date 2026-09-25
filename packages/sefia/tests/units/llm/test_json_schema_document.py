import pytest
from sefia.json_schema import JsonSchemaDocument, LocalDefinitionRef


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


def test_schema_package_has_no_llm_dependency() -> None:
    import subprocess
    import sys
    from pathlib import Path

    import sefia.json_schema as schema

    script = """
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("schema_boundary_test", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert not any(name == "sefia.llm" or name.startswith("sefia.llm.") for name in sys.modules)
assert not {"JsonValue", "JsonScalar", "JsonObject"} & set(module.__all__)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(Path(schema.__file__))],
        check=True,
    )
