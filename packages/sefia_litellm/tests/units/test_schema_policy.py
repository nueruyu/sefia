from copy import deepcopy
from typing import Any

import pytest
from sefia.llm.json_schema import JsonObject
from sefia_litellm._schema._policy import (
    GENERATED_SCHEMA_POLICY,
    USER_DEFINED_SCHEMA_POLICY,
    prepare_schema,
)


def test_compatible_raw_tool_schema_is_preserved_verbatim() -> None:
    raw_schema: JsonObject = {
        "title": "SearchArguments",
        "type": "object",
        "properties": {"query": {"title": "Query", "type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    schema = prepare_schema(
        deepcopy(raw_schema), USER_DEFINED_SCHEMA_POLICY
    ).wire_schema

    assert schema == raw_schema


@pytest.mark.parametrize(
    ("raw_schema", "message"),
    [
        (
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "additionalProperties": False,
            },
            "all object properties must be required",
        ),
        (
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "additionalProperties to false",
        ),
        (
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": True,
            },
            "additionalProperties to false",
        ),
        (
            {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "additionalProperties to false",
        ),
        (
            {
                "oneOf": [
                    {"type": "object", "additionalProperties": False},
                    {"type": "object", "additionalProperties": False},
                ]
            },
            "oneOf is not supported",
        ),
    ],
)
def test_incompatible_raw_tool_schema_is_rejected_without_rewriting(
    raw_schema: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_schema(deepcopy(raw_schema), USER_DEFINED_SCHEMA_POLICY)


_UNSUPPORTED_COMPOSITIONS: list[tuple[str, Any]] = [
    ("allOf", [{}]),
    ("not", {}),
    ("dependentRequired", {"query": ["other"]}),
    ("dependentSchemas", {"query": {}}),
    ("if", {}),
    ("then", {}),
    ("else", {}),
]


@pytest.mark.parametrize(
    ("keyword", "value"),
    _UNSUPPORTED_COMPOSITIONS,
)
def test_unsupported_composition_keyword_is_rejected(keyword: str, value: Any) -> None:
    raw_schema: JsonObject = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
        keyword: value,
    }

    with pytest.raises(ValueError, match=rf"{keyword} is not supported"):
        prepare_schema(deepcopy(raw_schema), USER_DEFINED_SCHEMA_POLICY)


@pytest.mark.parametrize(
    "property_name",
    [
        "allOf",
        "not",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
        "oneOf",
    ],
)
def test_schema_keyword_is_allowed_as_property_name(property_name: str) -> None:
    raw_schema: JsonObject = {
        "type": "object",
        "properties": {property_name: {"type": "string"}},
        "required": [property_name],
        "additionalProperties": False,
    }

    schema = prepare_schema(
        deepcopy(raw_schema), USER_DEFINED_SCHEMA_POLICY
    ).wire_schema

    assert schema == raw_schema


def test_generated_schema_is_corrected_recursively() -> None:
    schema: JsonObject = {
        "title": "Result",
        "type": "object",
        "properties": {
            "value": {
                "title": "Value",
                "oneOf": [{"type": "string"}, {"type": "integer"}],
            }
        },
    }
    prepared = prepare_schema(schema, GENERATED_SCHEMA_POLICY)
    assert prepared.wire_schema == {
        "type": "object",
        "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
        "required": ["value"],
        "additionalProperties": False,
    }
