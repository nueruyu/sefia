from typing import Any

from pydantic import TypeAdapter, ValidationError
from typing_extensions import final, override

from ..llm.json_schema import JsonSchemaDocument
from ..llm.result_schema import ResultSchema, ResultSchemaFactory
from ..llm.llm_output import LLMOutput


@final
class PydanticResultSchema(ResultSchema):
    def __init__(self, python_type: Any):
        self._adapter = TypeAdapter(python_type)
        self._json_schema = JsonSchemaDocument.from_mapping(self._adapter.json_schema())

    @property
    @override
    def json_schema(self) -> JsonSchemaDocument:
        return self._json_schema

    @override
    def validate(self, value: LLMOutput) -> Any:
        try:
            return self._adapter.validate_python(value.data)
        except ValidationError as error:
            raise ValueError(str(error)) from error


@final
class PydanticResultSchemaFactory(ResultSchemaFactory):
    @override
    def create(self, python_type: Any) -> ResultSchema:
        return PydanticResultSchema(python_type)
