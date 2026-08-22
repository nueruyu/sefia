from typing import Any

from pydantic import TypeAdapter, ValidationError
from typing_extensions import final, override

from ..llm.json_schema import JsonSchemaDocument
from ..llm.result_format import ResultFormat, ResultFormatFactory
from ..llm.llm_output import LLMOutput


@final
class PydanticResultFormat(ResultFormat):
    def __init__(self, python_type: Any):
        self._adapter = TypeAdapter(python_type)
        self._schema = JsonSchemaDocument.from_mapping(self._adapter.json_schema())

    @property
    @override
    def schema(self) -> JsonSchemaDocument:
        return self._schema

    @override
    def validate(self, value: LLMOutput) -> Any:
        try:
            return self._adapter.validate_python(value.data)
        except ValidationError as error:
            raise ValueError(str(error)) from error


@final
class PydanticResultFormatFactory(ResultFormatFactory):
    @override
    def create(self, python_type: Any) -> ResultFormat:
        return PydanticResultFormat(python_type)
