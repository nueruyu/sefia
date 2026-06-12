import json
from collections import deque
from typing import Any, Awaitable, Callable

from .._interfaces.middleware import StepContext, StepMiddleware
from ..exceptions import InferenceControlSignal
from ..inference import InferenceDecision, ToolCallDecision


class StagnationError(InferenceControlSignal):
    pass


class StagnationDetector(StepMiddleware):
    def __init__(self, max_repeats: int = 3):
        if max_repeats < 2:
            raise ValueError("max_repeats must be at least 2")
        self.max_repeats = max_repeats
        self.history: deque[str] = deque(maxlen=max_repeats)

    def _hash_tool_call(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        serialized_args = json.dumps(tool_args, sort_keys=True, default=str)
        return f"{tool_name}({serialized_args})"

    def _record_and_check(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        call_hash = self._hash_tool_call(tool_name, tool_args)
        self.history.append(call_hash)

        if len(self.history) == self.max_repeats and len(set(self.history)) == 1:
            raise StagnationError(
                f"Detected stagnation: Tool '{tool_name}' called {self.max_repeats} "
                "times with the same arguments."
            )

    async def wrap(
        self,
        ctx: StepContext,
        nxt: Callable[[], Awaitable[InferenceDecision]],
    ) -> InferenceDecision:
        if ctx.step == 0:
            self.history.clear()
        decision = await nxt()
        if isinstance(decision, ToolCallDecision):
            for call in decision.calls:
                self._record_and_check(call.name, call.arguments)
        return decision
