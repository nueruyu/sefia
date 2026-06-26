from dataclasses import dataclass

import pytest

from sefia.pydantic import PydanticModelInspector


@dataclass(frozen=True)
class _Item:
    name: str
    count: int = 0


def _sample_func(a: int, b: str = "x") -> bool:
    """Sample function."""
    return True


class TestPydanticModelInspector:
    def test_get_schema_for_primitive_type(self):
        inspector = PydanticModelInspector()

        schema = inspector.get_type_schema(str)

        assert schema["type"] == "string"

    def test_get_schema_for_dataclass_type(self):
        inspector = PydanticModelInspector()

        schema = inspector.get_type_schema(_Item)

        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]

    def test_get_function_schema(self):
        inspector = PydanticModelInspector()

        schema = inspector.get_function_schema(_sample_func)

        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "_sample_func"
        assert fn["description"] == "Sample function."
        assert fn["parameters"]["properties"]["a"]["type"] == "integer"
        assert fn["parameters"]["properties"]["b"]["type"] == "string"
        assert fn["parameters"]["properties"]["b"]["default"] == "x"
        assert fn["parameters"]["required"] == ["a"]
        assert fn["parameters"]["additionalProperties"] is False

    def test_get_function_name_sanitizes_complex_names(self):
        class Outer:
            class Inner:
                def my_method(self):
                    pass

        name = PydanticModelInspector().get_function_name(Outer.Inner.my_method)

        assert name.endswith("Outer_Inner_my_method")
        assert "." not in name
        assert "<" not in name

    def test_get_function_schema_is_cached(self):
        inspector = PydanticModelInspector()

        schema1 = inspector.get_function_schema(_sample_func)
        schema2 = inspector.get_function_schema(_sample_func)

        assert schema1 is schema2

    def test_validate_dataclass(self):
        inspector = PydanticModelInspector()

        obj = inspector.validate(_Item, {"name": "book", "count": 2})

        assert isinstance(obj, _Item)
        assert obj.name == "book"
        assert obj.count == 2

    def test_validate_raises_on_invalid_data(self):
        inspector = PydanticModelInspector()

        with pytest.raises(ValueError, match="Model validation failed"):
            inspector.validate(_Item, {"name": "book", "count": "bad"})
