from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from types import MappingProxyType
from uuid import UUID

import pytest
from pydantic import BaseModel
from sefia.pydantic import PydanticStructuredDataConverter


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
def test_to_structured_data_normalizes_primitives(
    value: object, expected: object
) -> None:
    converter = PydanticStructuredDataConverter()

    assert converter.to_structured_data(value).tree == expected


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

    assert PydanticStructuredDataConverter().to_structured_data(value).tree == {
        str(identifier): {
            "record": {"status": "ready", "created": "2026-09-23"},
            "model": {"value": 3},
            "timestamp": "2026-09-23T10:30:00",
        }
    }


def test_to_structured_data_rejects_mapping_key_collisions() -> None:
    identifier = UUID("12345678-1234-5678-1234-567812345678")

    with pytest.raises(ValueError, match="same structured key"):
        PydanticStructuredDataConverter().to_structured_data(
            {identifier: "first", str(identifier): "second"}
        )


def test_to_structured_data_falls_back_to_string_for_unknown_values() -> None:
    data = PydanticStructuredDataConverter().to_structured_data(_Unknown())

    assert data.tree == "fallback"
