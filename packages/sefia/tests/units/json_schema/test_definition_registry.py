from typing import Any

import pytest
from sefia.json_schema import DefinitionRegistry


def test_definition_registry_imports_definitions_and_rewrites_references() -> None:
    definitions: dict[str, Any] = {}
    registry = DefinitionRegistry(definitions)
    fragment: dict[str, Any] = {
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
    definitions: dict[str, Any] = {}
    registry = DefinitionRegistry(definitions)
    first: dict[str, Any] = {
        "$ref": "#/$defs/A",
        "$defs": {
            "A": {"$ref": "#/$defs/B"},
            "B": {"$ref": "#/$defs/C"},
            "C": {"type": "string"},
        },
    }
    second: dict[str, Any] = {
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
    definitions: dict[str, Any] = {}
    registry = DefinitionRegistry(definitions)
    fragment: dict[str, Any] = {
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
    definitions: dict[str, Any] = {}
    registry = DefinitionRegistry(definitions)
    fragment: dict[str, Any] = {
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
    definitions: dict[str, Any] = {}
    registry = DefinitionRegistry(definitions)
    fragment: dict[str, Any] = {
        "$ref": "#/definitions/Item",
        "definitions": {"Item": {"type": "string"}},
    }

    imported = registry.import_schema(fragment, namespace="legacy")

    assert imported == {"$ref": "#/$defs/legacy__Item"}
    assert definitions == {"legacy__Item": {"type": "string"}}


def test_definition_registry_rejects_overlapping_definition_keywords() -> None:
    registry = DefinitionRegistry({})
    fragment: dict[str, Any] = {
        "type": "object",
        "properties": {
            "modern": {"$ref": "#/$defs/Item"},
            "legacy": {"$ref": "#/definitions/Item"},
        },
        "$defs": {"Item": {"type": "string"}},
        "definitions": {"Item": {"type": "integer"}},
    }

    with pytest.raises(
        ValueError,
        match=r"\$defs and definitions contain the same names: \['Item'\]",
    ):
        registry.import_schema(fragment, namespace="fragment")

    assert fragment["$defs"] == {"Item": {"type": "string"}}
    assert fragment["definitions"] == {"Item": {"type": "integer"}}


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
    fragment: dict[str, Any] = {"$ref": reference}

    with pytest.raises(ValueError, match=message):
        registry.import_schema(fragment, namespace="fragment")
