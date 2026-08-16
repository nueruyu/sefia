from typing import Any

from pydantic import TypeAdapter, ValidationError
from typing_extensions import final, override

from ..llm.json_schema import JsonSchemaDocument
from ..llm.structured_output import (
    StructuredValue,
    StructuredValueSchema,
    StructuredValueSchemaFactory,
)


@final
class PydanticStructuredValueSchema(StructuredValueSchema):
    def __init__(self, python_type: Any):
        self._adapter = TypeAdapter(python_type)
        self._json_schema = JsonSchemaDocument.from_mapping(self._adapter.json_schema())

    @property
    @override
    def json_schema(self) -> JsonSchemaDocument:
        return self._json_schema

    @override
    def validate(self, value: StructuredValue) -> Any:
        try:
            return self._adapter.validate_python(value)
        except ValidationError as error:
            raise ValueError(str(error)) from error


@final
class PydanticStructuredValueSchemaFactory(StructuredValueSchemaFactory):
    @override
    def create(self, python_type: Any) -> StructuredValueSchema:
        return PydanticStructuredValueSchema(python_type)
