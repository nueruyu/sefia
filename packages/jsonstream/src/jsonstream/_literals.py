import json
from typing import NoReturn

from .events import JsonScalar


def decode_literal(literal: str) -> JsonScalar:
    try:
        value = json.loads(literal, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid literal or number: {literal}") from error

    if isinstance(value, (dict, list, str)):
        raise ValueError(f"Invalid literal or number: {literal}")

    return value


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"Invalid JSON constant: {value}")
