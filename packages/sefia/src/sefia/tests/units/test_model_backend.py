import pytest

from sefia.pydantic import PydanticModelBackend
from sefia.pydantic._function_models import PydanticFunctionModelFactory


def _sample_func(a: int, b: str = "x") -> bool:
    """Sample function."""
    return True


def _positional_only_func(value: int, /) -> bool:
    return True


class TestPydanticModelBackend:
    def test_get_function_schema(self):
        backend = PydanticModelBackend()

        schema = backend.get_function_schema(_sample_func)

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

        name = PydanticModelBackend().get_function_name(Outer.Inner.my_method)

        assert name.endswith("Outer_Inner_my_method")
        assert "." not in name
        assert "<" not in name

    def test_get_function_schema_is_cached(self):
        backend = PydanticModelBackend()

        schema1 = backend.get_function_schema(_sample_func)
        schema2 = backend.get_function_schema(_sample_func)

        assert schema1 is schema2

    def test_get_function_schema_rejects_positional_only_parameters(self):
        backend = PydanticModelBackend()

        with pytest.raises(ValueError, match="positional-only.*keyword arguments"):
            backend.get_function_schema(_positional_only_func)

    def test_function_model_factory_reuses_params_model(self):
        factory = PydanticFunctionModelFactory()

        model1 = factory.params_model(
            _sample_func,
            name="_sample_func",
            forbid_extra=True,
        )
        model2 = factory.params_model(
            _sample_func,
            name="_sample_func",
            forbid_extra=True,
        )

        assert model1 is model2
