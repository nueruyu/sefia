from dataclasses import dataclass

import pytest

from sefia.llm import StructuredData
from sefia.json_schema import SchemaNode
from sefia.pydantic import PydanticResultFormatFactory


@dataclass(frozen=True)
class _Result:
    value: int


def test_result_format_exposes_schema_and_restores_python_value() -> None:
    result_format = PydanticResultFormatFactory().create(_Result)

    schema = SchemaNode(result_format.schema.to_dict())
    assert schema.properties()["value"].type == "integer"
    assert result_format.validate(StructuredData.from_json({"value": 3})) == _Result(
        value=3
    )


def test_result_format_rejects_invalid_structured_data() -> None:
    result_format = PydanticResultFormatFactory().create(_Result)

    with pytest.raises(ValueError):
        result_format.validate(StructuredData.from_json({"value": "invalid"}))
