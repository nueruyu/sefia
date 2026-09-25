from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..inference import FunctionInfo
from ._messages import Message


@dataclass(frozen=True)
class MessageLayout:
    before: tuple[Message, ...]
    arguments: Mapping[str, Any]
    after: tuple[Message, ...]

    @classmethod
    def default(cls, function: FunctionInfo) -> MessageLayout:
        return cls(before=(), arguments=dict(function.prompt_arguments), after=())
