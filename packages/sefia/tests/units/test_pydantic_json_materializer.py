from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from types import MappingProxyType
from uuid import UUID

import pytest
from pydantic import BaseModel
from sefia.pydantic import PydanticJsonMaterializer


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
def test_materialize_normalizes_primitives(value: object, expected: object) -> None:
    converter = PydanticJsonMaterializer()

    assert converter.materialize(value) == expected


def test_materialize_normalizes_nested_application_values() -> None:
    identifier = UUID("12345678-1234-5678-1234-567812345678")
    value = MappingProxyType(
        {
            "entry": {
                "record": _Record(_Status.READY, date(2026, 9, 23)),
                "model": _Model(value=3),
                "timestamp": datetime(2026, 9, 23, 10, 30),
                "id": identifier,
            }
        }
    )

    assert PydanticJsonMaterializer().materialize(value) == {
        "entry": {
            "record": {"status": "ready", "created": "2026-09-23"},
            "model": {"value": 3},
            "timestamp": "2026-09-23T10:30:00",
            "id": str(identifier),
        }
    }


def test_materialize_rejects_mapping_key_collisions() -> None:
    identifier = UUID("12345678-1234-5678-1234-567812345678")

    with pytest.raises(ValueError, match="JSON object keys must be strings"):
        PydanticJsonMaterializer().materialize(
            {identifier: "first", str(identifier): "second"}
        )


def test_materialize_falls_back_to_string_for_unknown_values() -> None:
    data = PydanticJsonMaterializer().materialize(_Unknown())

    assert data == "fallback"


@pytest.mark.parametrize("key", [1, None, UUID(int=0)])
def test_rejects_non_string_keys_at_any_depth(key: object) -> None:
    with pytest.raises(ValueError, match="JSON object keys must be strings"):
        PydanticJsonMaterializer().materialize({"nested": [{key: "value"}]})


def test_materialized_containers_are_detached() -> None:
    source = {"items": [1]}
    result = PydanticJsonMaterializer().materialize(source)
    source["items"].append(2)
    assert result == {"items": [1]}


def test_model_dump_must_not_stringify_application_mapping_keys() -> None:
    class Model(BaseModel):
        entries: dict[int, str]

    with pytest.raises(ValueError, match="JSON object keys must be strings"):
        PydanticJsonMaterializer().materialize(Model(entries={1: "one"}))
