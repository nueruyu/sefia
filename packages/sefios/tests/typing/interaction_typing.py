"""Pyright checks return inference for public result type expressions."""

from typing import Annotated, Literal, assert_type

from pydantic import BaseModel, Field
from sefios import require_interaction
from typing_extensions import TypedDict


class WeatherResult(BaseModel):
    temperature: int


class Approval(TypedDict):
    approved: bool


async def result_type_forms() -> None:
    assert_type(await require_interaction("str", None, str), str)
    assert_type(await require_interaction("list", None, list[int]), list[int])
    assert_type(
        await require_interaction("optional", None, WeatherResult | None),
        WeatherResult | None,
    )
    assert_type(
        await require_interaction("dict", None, dict[str, WeatherResult]),
        dict[str, WeatherResult],
    )
    assert_type(
        await require_interaction("literal", None, Literal["yes", "no"]),
        Literal["yes", "no"],
    )
    assert_type(await require_interaction("typed-dict", None, Approval), Approval)
    assert_type(
        await require_interaction(
            "annotated", None, Annotated[str, Field(min_length=1)]
        ),
        str,
    )
