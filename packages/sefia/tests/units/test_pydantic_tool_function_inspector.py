from dataclasses import dataclass

import pytest
from pydantic import BaseModel
from sefia.pydantic import PydanticToolFunctionInspector


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


def test_definition() -> None:
    inspector = PydanticToolFunctionInspector()

    definition = inspector.definition(_sample_func, name="_sample_func")

    assert definition.name == "_sample_func"
    assert definition.description == "Sample function."
    params = definition.parameters
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["type"] == "string"
    assert params["properties"]["b"]["default"] == "x"
    assert params["required"] == ["a"]
    assert params["additionalProperties"] is False


def test_tool_name_sanitizes_complex_names() -> None:
    class Outer:
        class Inner:
            def my_method(self) -> None:
                pass

    name = PydanticToolFunctionInspector().tool_name(Outer.Inner.my_method)

    assert name.endswith("Outer_Inner_my_method")
    assert "." not in name
    assert "<" not in name


def test_definition_is_cached() -> None:
    inspector = PydanticToolFunctionInspector()

    definition1 = inspector.definition(_sample_func, name="_sample_func")
    definition2 = inspector.definition(_sample_func, name="_sample_func")

    assert definition1 is definition2


def test_definition_caches_unhashable_callables() -> None:
    inspector = PydanticToolFunctionInspector()
    func = _UnhashableCallable()

    definition1 = inspector.definition(func, name="unhashable")
    definition2 = inspector.definition(func, name="unhashable")

    assert definition1 is definition2


def test_definition_rejects_positional_only_parameters() -> None:
    inspector = PydanticToolFunctionInspector()

    with pytest.raises(ValueError, match="positional-only.*keyword arguments"):
        inspector.definition(_positional_only_func, name="_positional_only_func")


def test_bind_coerces_and_passes_extra_keys_through() -> None:
    inspector = PydanticToolFunctionInspector()

    assert inspector.bind(_sample_func, {"a": "1", "b": "y"}) == {
        "a": 1,
        "b": "y",
    }


def test_bind_preserves_coerced_model_and_dataclass_instances() -> None:
    class Point(BaseModel):
        x: int
        y: int

    @dataclass
    class Box:
        width: int

    def func(point: Point, box: Box) -> None: ...

    bound = PydanticToolFunctionInspector().bind(
        func, {"point": {"x": 1, "y": 2}, "box": {"width": 3}}
    )

    assert bound["point"] == Point(x=1, y=2)
    assert isinstance(bound["box"], Box)
    assert bound["box"].width == 3
