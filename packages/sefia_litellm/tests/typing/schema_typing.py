from sefia_litellm._schema._types import SchemaObject
from sefia_litellm._schema._policy import GENERATED_SCHEMA_POLICY, prepare_schema


def schema_values_must_be_json_compatible() -> None:
    schema: SchemaObject = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    prepare_schema(schema, GENERATED_SCHEMA_POLICY)

    invalid: SchemaObject = {"default": object()}  # pyright: ignore[reportAssignmentType]
    prepare_schema(invalid, GENERATED_SCHEMA_POLICY)
    prepare_schema({"default": object()}, GENERATED_SCHEMA_POLICY)  # pyright: ignore[reportArgumentType]
