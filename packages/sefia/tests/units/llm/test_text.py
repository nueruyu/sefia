import json
from dataclasses import dataclass
from datetime import date
from enum import Enum
from types import MappingProxyType
from uuid import UUID

import pytest
from pydantic import BaseModel

from sefia.llm._text import TextFormatter
from sefia.pydantic._json_utils import pydantic_json_default


class _Status(Enum):
    READY = "ready"


@dataclass(frozen=True)
class _Record:
    status: _Status
    created: date


class _Model(BaseModel):
    value: int


@pytest.mark.parametrize(
    "formatter", [TextFormatter(), TextFormatter(pydantic_json_default)]
)
def test_json_text_preserves_nested_python_values(formatter: TextFormatter) -> None:
    identifier = UUID("12345678-1234-5678-1234-567812345678")
    value = MappingProxyType(
        {
            identifier: {
                "record": _Record(_Status.READY, date(2026, 9, 21)),
                "model": _Model(value=3),
            }
        }
    )

    rendered = formatter.compact_json(value)

    assert json.loads(rendered) == {
        str(identifier): {
            "record": {"status": "ready", "created": "2026-09-21"},
            "model": {"value": 3},
        }
    }
