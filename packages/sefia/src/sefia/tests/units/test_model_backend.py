import gc
import weakref
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from sefia.pydantic import PydanticModelBackend
from sefia.pydantic._function_models import PydanticFunctionModelFactory, cache_key


def _sample_func(a: int, b: str = "x") -> bool:
    """Sample function."""
    return True


def _positional_only_func(value: int, /) -> bool:
    return True


class _UnhashableCallable:
    def __call__(self, value: int) -> str:
        return str(value)

    def __eq__(self, other: object) -> bool:
        return self is other


class TestPydanticModelBackend:
    def test_definition(self):
        backend = PydanticModelBackend()

        definition = backend.definition(_sample_func, name="_sample_func")

        assert definition.name == "_sample_func"
        assert definition.description == "Sample function."
        params = definition.parameters
        assert params["properties"]["a"]["type"] == "integer"
        assert params["properties"]["b"]["type"] == "string"
        assert params["properties"]["b"]["default"] == "x"
        assert params["required"] == ["a"]
        assert params["additionalProperties"] is False

    def test_tool_name_sanitizes_complex_names(self):
        class Outer:
            class Inner:
                def my_method(self):
                    pass

        name = PydanticModelBackend().tool_name(Outer.Inner.my_method)

        assert name.endswith("Outer_Inner_my_method")
        assert "." not in name
        assert "<" not in name

    def test_definition_is_cached(self):
        backend = PydanticModelBackend()

        definition1 = backend.definition(_sample_func, name="_sample_func")
        definition2 = backend.definition(_sample_func, name="_sample_func")

        assert definition1 is definition2

    def test_definition_caches_unhashable_callables(self):
        backend = PydanticModelBackend()
        func = _UnhashableCallable()

        definition1 = backend.definition(func, name="unhashable")
        definition2 = backend.definition(func, name="unhashable")

        assert definition1 is definition2

    def test_cache_key_uses_identity_for_unhashable_callables(self):
        func = _UnhashableCallable()

        key1 = cache_key(func)
        key2 = cache_key(func)
        other_key = cache_key(_UnhashableCallable())

        assert key1 == key2
        assert hash(key1) == hash(key2)
        assert key1 != other_key

    def test_cache_key_keeps_unhashable_callables_alive(self):
        func = _UnhashableCallable()
        ref = weakref.ref(func)
        key = cache_key(func)

        del func
        gc.collect()

        assert key is not None
        assert ref() is not None

    def test_definition_rejects_positional_only_parameters(self):
        backend = PydanticModelBackend()

        with pytest.raises(ValueError, match="positional-only.*keyword arguments"):
            backend.definition(_positional_only_func, name="_positional_only_func")

    def test_bind_coerces_and_passes_extra_keys_through(self):
        backend = PydanticModelBackend()

        # Declared params are coerced; the shape is enforced upstream, so bind
        # passes any additional keys through unchanged.
        assert backend.bind(_sample_func, {"a": "1", "b": "y"}) == {"a": 1, "b": "y"}

    def test_bind_preserves_coerced_model_and_dataclass_instances(self):
        class Point(BaseModel):
            x: int
            y: int

        @dataclass
        class Box:
            width: int

        def func(point: Point, box: Box) -> None: ...

        bound = PydanticModelBackend().bind(
            func, {"point": {"x": 1, "y": 2}, "box": {"width": 3}}
        )

        # The coerced instances survive binding — a recursive dump would
        # flatten them back into dicts.
        assert bound["point"] == Point(x=1, y=2)
        assert isinstance(bound["box"], Box)
        assert bound["box"].width == 3

    def test_function_model_factory_reuses_params_model(self):
        factory = PydanticFunctionModelFactory()

        model1 = factory.params_model(
            _sample_func,
            name="_sample_func",
            extra="forbid",
        )
        model2 = factory.params_model(
            _sample_func,
            name="_sample_func",
            extra="forbid",
        )

        assert model1 is model2
