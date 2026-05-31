import json
from collections import deque
from typing import Any, Type

from sefia.events import BeforeToolCall, Event
from sefia.interfaces import EventHandler


class StagnationError(Exception):
    """Raised when the inference process appears to be stuck in a loop."""


class StagnationDetector(EventHandler[BeforeToolCall]):
    """
    Detects if the agent is stagnating by repeating the same tool calls.
    """

    def __init__(self, max_repeats: int = 3):
        self.max_repeats = max_repeats
        self.history: deque[str] = deque(maxlen=max_repeats)

    @property
    def event_types(self) -> tuple[Type[Event], ...]:
        return (BeforeToolCall,)

    def _hash_tool_call(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        serialized_args = json.dumps(tool_args, sort_keys=True, default=str)
        return f"{tool_name}({serialized_args})"

    def _record_and_check(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        """Records a tool call and checks for stagnation."""
        call_hash = self._hash_tool_call(tool_name, tool_args)
        self.history.append(call_hash)

        if len(self.history) == self.max_repeats and len(set(self.history)) == 1:
            raise StagnationError(
                f"Detected stagnation: Tool '{tool_name}' called {self.max_repeats} "
                "times with the same arguments."
            )

    async def handle(self, event: BeforeToolCall) -> None:
        tool_call = event.tool_call
        self._record_and_check(tool_call.name, tool_call.arguments)
