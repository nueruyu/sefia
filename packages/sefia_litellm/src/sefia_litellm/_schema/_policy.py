from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import final

from sefia.llm.json_schema import (
    JsonObject,
    SchemaKeyword,
    SchemaNode,
    SchemaPath,
    without_titles,
)

from ._uniform_dictionary import UniformDictionaryFormat

K = SchemaKeyword


class MissingFieldAction(StrEnum):
    ADD = "add"
    REJECT = "reject"


class OneOfAction(StrEnum):
    REWRITE_AS_ANY_OF = "rewrite_as_any_of"
    REJECT = "reject"


class MappingAction(StrEnum):
    LOWER_TO_ENTRIES = "lower_to_entries"
    REJECT = "reject"


@final
@dataclass(frozen=True)
class SchemaConstraints:
    objects_must_be_closed: bool
    all_properties_must_be_required: bool
    unsupported_keywords: frozenset[SchemaKeyword]


@final
@dataclass(frozen=True)
class SchemaPolicy:
    constraints: SchemaConstraints
    missing_additional_properties: MissingFieldAction
    missing_required_properties: MissingFieldAction
    one_of: OneOfAction
    mappings: MappingAction
    strip_titles: bool


STRICT_OUTPUT_CONSTRAINTS = SchemaConstraints(
    objects_must_be_closed=True,
    all_properties_must_be_required=True,
    unsupported_keywords=frozenset(
        {
            K.ALL_OF,
            K.NOT,
            K.DEPENDENT_REQUIRED,
            K.DEPENDENT_SCHEMAS,
            K.IF,
            K.THEN,
            K.ELSE,
        }
    ),
)

GENERATED_SCHEMA_POLICY = SchemaPolicy(
    constraints=STRICT_OUTPUT_CONSTRAINTS,
    missing_additional_properties=MissingFieldAction.ADD,
    missing_required_properties=MissingFieldAction.ADD,
    one_of=OneOfAction.REWRITE_AS_ANY_OF,
    mappings=MappingAction.LOWER_TO_ENTRIES,
    strip_titles=True,
)

USER_DEFINED_SCHEMA_POLICY = SchemaPolicy(
    constraints=STRICT_OUTPUT_CONSTRAINTS,
    missing_additional_properties=MissingFieldAction.REJECT,
    missing_required_properties=MissingFieldAction.REJECT,
    one_of=OneOfAction.REJECT,
    mappings=MappingAction.REJECT,
    strip_titles=False,
)


@final
@dataclass(frozen=True)
class PreparedSchema:
    wire_schema: JsonObject
    dictionary_format: UniformDictionaryFormat | None


def prepare_schema(schema: JsonObject, policy: SchemaPolicy) -> PreparedSchema:
    _apply_corrections(schema, policy)
    dictionary_format = (
        UniformDictionaryFormat.from_schema(schema)
        if policy.mappings is MappingAction.LOWER_TO_ENTRIES
        else None
    )
    _validate(schema, policy.constraints)
    return PreparedSchema(schema, dictionary_format)


def _apply_corrections(schema: JsonObject, policy: SchemaPolicy) -> None:
    if policy.strip_titles:
        stripped = without_titles(schema)
        assert isinstance(stripped, dict)
        schema.clear()
        schema.update(stripped)
    for cursor in SchemaNode(schema).walk():
        node = cursor.node
        if node.type == "object":
            _apply_object_corrections(node, policy)
        if policy.one_of is OneOfAction.REWRITE_AS_ANY_OF:
            alternatives = node.value.pop(K.ONE_OF, None)
            if alternatives is not None:
                node.value[K.ANY_OF] = alternatives


def _apply_object_corrections(node: SchemaNode, policy: SchemaPolicy) -> None:
    if policy.missing_additional_properties is MissingFieldAction.ADD:
        node.value.setdefault(K.ADDITIONAL_PROPERTIES, False)
    properties = node.object_map(K.PROPERTIES)
    if (
        properties is not None
        and policy.missing_required_properties is MissingFieldAction.ADD
    ):
        node.value[K.REQUIRED] = list(properties)


def _validate(schema: JsonObject, constraints: SchemaConstraints) -> None:
    for cursor in SchemaNode(schema).walk():
        path, node = cursor.path, cursor.node
        if K.ONE_OF in node.value:
            _unsupported(path, "oneOf is not supported; use a disjoint anyOf")
        for keyword in constraints.unsupported_keywords:
            if keyword in node.value:
                _unsupported(path, f"{keyword} is not supported")
        if node.type == "object":
            _validate_object(path, node, constraints)


def _validate_object(
    path: SchemaPath, node: SchemaNode, constraints: SchemaConstraints
) -> None:
    if constraints.objects_must_be_closed and node.additional_properties() is not False:
        _unsupported(path, "object schemas must set additionalProperties to false")
    if not constraints.all_properties_must_be_required:
        return
    properties = node.properties()
    required_names = set(node.required() or ())
    missing = sorted(set(properties) - required_names)
    if missing:
        _unsupported(path, f"all object properties must be required; missing {missing}")


def _unsupported(path: SchemaPath, detail: str) -> None:
    location = "/".join(map(str, path)) or "<root>"
    raise ValueError(
        "LLM schema is not compatible with strict structured output at "
        f"{location}: {detail}"
    )
