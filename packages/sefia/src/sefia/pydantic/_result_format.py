from typing import Any

from pydantic import TypeAdapter, ValidationError
from typing_extensions import final, override

from ..llm.json_schema import JsonSchemaDocument
from ..llm.result_format import ResultFormat, ResultFormatFactory
from ..llm.structured_data import StructuredData


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
    def validate(self, data: StructuredData) -> Any:
        try:
            return self._adapter.validate_python(data.tree)
        except ValidationError as error:
            raise ValueError(str(error)) from error


@final
class PydanticResultFormatFactory(ResultFormatFactory):
    @override
    def create(self, python_type: Any) -> ResultFormat:
        return PydanticResultFormat(python_type)
