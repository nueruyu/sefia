from copy import deepcopy

import pytest
from sefia.llm.json_schema import JsonObject
from sefia.llm.structured_data import StructuredData, StructuredDataTree
from sefia_litellm._schema._uniform_dictionary import UniformDictionaryFormat


def _mapping(value: JsonObject) -> JsonObject:
    return {"type": "object", "additionalProperties": value}


def _object(**properties: JsonObject) -> JsonObject:
    return {
        "type": "object",
        "properties": {name: value for name, value in properties.items()},
        "required": list(properties),
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    ("schema", "logical", "wire"),
    [
        (
            _mapping({"type": "string"}),
            {"Maintainability": "good", "Dependencies": "current"},
            [
                {"key": "Maintainability", "value": "good"},
                {"key": "Dependencies", "value": "current"},
            ],
        ),
        (
            _mapping(_mapping({"type": "integer"})),
            {"outer": {"inner": 1}},
            [{"key": "outer", "value": [{"key": "inner", "value": 1}]}],
        ),
        (
            _mapping(_mapping(_mapping({"type": "integer"}))),
            {"outer": {"middle": {"inner": 1}}},
            [
                {
                    "key": "outer",
                    "value": [
                        {"key": "middle", "value": [{"key": "inner", "value": 1}]}
                    ],
                }
            ],
        ),
        (
            {"type": "array", "items": _mapping({"type": "integer"})},
            [{"count": 3}],
            [[{"key": "count", "value": 3}]],
        ),
        (
            _object(values=_mapping(_mapping({"type": "integer"}))),
            {"values": {"outer": {"inner": 1}}},
            {"values": [{"key": "outer", "value": [{"key": "inner", "value": 1}]}]},
        ),
        (
            {
                "anyOf": [
                    _object(labels=_mapping({"type": "integer"})),
                    _object(text={"type": "string"}),
                ]
            },
            {"labels": {"important": 2}},
            {"labels": [{"key": "important", "value": 2}]},
        ),
        (
            {
                "anyOf": [
                    _object(x=_mapping({"type": "integer"})),
                    _object(
                        x={"type": "array", "items": {"type": "integer"}},
                        y=_mapping({"type": "integer"}),
                    ),
                ]
            },
            {"x": [1], "y": {"n": 2}},
            {"x": [1], "y": [{"key": "n", "value": 2}]},
        ),
        (
            {
                "$ref": "#/$defs/Report",
                "$defs": {
                    "Report": _object(
                        issues=_mapping(
                            {"type": "array", "items": {"$ref": "#/$defs/Issue"}}
                        )
                    ),
                    "Issue": _object(description={"type": "string"}),
                },
            },
            {"issues": {"style": [{"description": "Clearer names"}]}},
            {"issues": [{"key": "style", "value": [{"description": "Clearer names"}]}]},
        ),
    ],
    ids=[
        "mapping",
        "nested",
        "three-levels",
        "list",
        "object",
        "union",
        "fully-valid-union-branch",
        "references",
    ],
)
def test_mapping_format_round_trip(
    schema: JsonObject, logical: StructuredDataTree, wire: StructuredDataTree
) -> None:
    data_format = UniformDictionaryFormat.from_schema(deepcopy(schema))

    assert data_format.encode(StructuredData.from_tree(logical)).tree == wire
    assert data_format.decode(StructuredData.from_tree(wire)).tree == logical


def test_entry_schema_preserves_mapping_constraints() -> None:
    data_format = UniformDictionaryFormat.from_schema(
        {
            **_mapping({"type": "string"}),
            "minProperties": 1,
            "maxProperties": 2,
            "title": "Labels",
            "description": "Tags",
            "propertyNames": {"type": "string", "minLength": 1},
        }
    )

    assert data_format.schema == {
        "type": "array",
        "minItems": 1,
        "maxItems": 2,
        "title": "Labels",
        "description": "Tags",
        "items": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "minLength": 1},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        },
    }


def test_hybrid_object_is_not_lowered_as_a_dictionary() -> None:
    with pytest.raises(
        ValueError, match="objects combining fixed properties with dictionary values"
    ):
        UniformDictionaryFormat.from_schema(
            {
                **_mapping({"type": "integer"}),
                "properties": {"fixed": {"type": "string"}},
            }
        )


@pytest.mark.parametrize(
    ("wire", "message"),
    [
        (
            [{"key": "same", "value": "first"}, {"key": "same", "value": "second"}],
            "duplicate mapping key",
        ),
        ([{"key": "missing-value"}], "contain only key and value"),
    ],
)
def test_mapping_restoration_rejects_invalid_entries(
    wire: StructuredDataTree, message: str
) -> None:
    data_format = UniformDictionaryFormat.from_schema(_mapping({"type": "string"}))
    with pytest.raises(ValueError, match=message):
        data_format.decode(StructuredData.from_tree(wire))
