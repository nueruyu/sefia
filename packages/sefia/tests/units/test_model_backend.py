from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from types import MappingProxyType
from uuid import UUID

import pytest
from pydantic import BaseModel
from sefia.pydantic import PydanticModelBackend


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


class _Status(Enum):
    READY = "ready"


@dataclass(frozen=True)
class _Record:
    status: _Status
    created: date


class _Model(BaseModel):
    value: int


class _Unknown:
    def __str__(self) -> str:
        return "fallback"


def test_definition():
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


def test_tool_name_sanitizes_complex_names():
    class Outer:
        class Inner:
            def my_method(
                self,
            ):
                pass

    name = PydanticModelBackend().tool_name(Outer.Inner.my_method)

    assert name.endswith("Outer_Inner_my_method")
    assert "." not in name
    assert "<" not in name


def test_definition_is_cached():
    backend = PydanticModelBackend()

    definition1 = backend.definition(_sample_func, name="_sample_func")
    definition2 = backend.definition(_sample_func, name="_sample_func")

    assert definition1 is definition2


def test_definition_caches_unhashable_callables():
    backend = PydanticModelBackend()
    func = _UnhashableCallable()

    definition1 = backend.definition(func, name="unhashable")
    definition2 = backend.definition(func, name="unhashable")

    assert definition1 is definition2


def test_definition_rejects_positional_only_parameters():
    backend = PydanticModelBackend()

    with pytest.raises(ValueError, match="positional-only.*keyword arguments"):
        backend.definition(_positional_only_func, name="_positional_only_func")


def test_bind_coerces_and_passes_extra_keys_through():
    backend = PydanticModelBackend()

    # Declared params are coerced; the shape is enforced upstream, so bind
    # passes any additional keys through unchanged.
    assert backend.bind(_sample_func, {"a": "1", "b": "y"}) == {"a": 1, "b": "y"}


def test_bind_preserves_coerced_model_and_dataclass_instances():
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("text", "text"),
        (1, 1),
        (1.5, 1.5),
        (True, True),
        ([1, "two"], [1, "two"]),
        ((1, "two"), [1, "two"]),
    ],
)
def test_to_structured_data_normalizes_primitives(
    value: object, expected: object
) -> None:
    assert PydanticModelBackend().to_structured_data(value).tree == expected


def test_to_structured_data_normalizes_nested_application_values() -> None:
    identifier = UUID("12345678-1234-5678-1234-567812345678")
    value = MappingProxyType(
        {
            identifier: {
                "record": _Record(_Status.READY, date(2026, 9, 23)),
                "model": _Model(value=3),
                "timestamp": datetime(2026, 9, 23, 10, 30),
            }
        }
    )

    assert PydanticModelBackend().to_structured_data(value).tree == {
        str(identifier): {
            "record": {"status": "ready", "created": "2026-09-23"},
            "model": {"value": 3},
            "timestamp": "2026-09-23T10:30:00",
        }
    }


def test_to_structured_data_rejects_mapping_key_collisions() -> None:
    identifier = UUID("12345678-1234-5678-1234-567812345678")

    with pytest.raises(ValueError, match="same structured key"):
        PydanticModelBackend().to_structured_data(
            {identifier: "first", str(identifier): "second"}
        )


def test_to_structured_data_falls_back_to_string_for_unknown_values() -> None:
    assert PydanticModelBackend().to_structured_data(_Unknown()).tree == "fallback"
