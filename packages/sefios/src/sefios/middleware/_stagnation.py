import json
from collections import deque
from typing import Any, Awaitable, Callable

from sefia._interfaces.middleware import StepContext, StepMiddleware
from sefia.exceptions import SefiaError
from sefia.inference import InferenceDecision, ToolCallDecision
from typing_extensions import final, override


class StagnationError(SefiaError):
    """Raised when the inference run appears stuck repeating the same tool call."""


@final
class StagnationDetector(StepMiddleware):
    """
    Detects if the agent is stagnating by repeating the same tool call.

    The middleware inspects each step's decision and records its tool calls. If
    the same call recurs ``max_repeats`` times in a row it raises
    ``StagnationError`` before the repeated tool runs again.

    The rolling window is kept on the instance. Middleware is instantiated per
    inference run (``Policy.create_middleware`` is called once per ``@infer``
    invocation in ``decorators._run``), so an instance is never shared across
    concurrent runs. A resume or retry continues the durable run from its saved
    step rather than replaying from the start, so the window keeps tracking the
    real sequence of consecutive tool calls across that boundary.
    """

    def __init__(self, max_repeats: int = 3):
        # Stagnation requires repetition, so a limit of 1 would flag the very
        # first tool call. The smallest meaningful value is 2.
        if max_repeats < 2:
            raise ValueError("max_repeats must be at least 2")
        self.max_repeats = max_repeats
        self.history: deque[str] = deque(maxlen=max_repeats)

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

    @override
    async def wrap(
        self,
        ctx: StepContext,
        nxt: Callable[[], Awaitable[InferenceDecision]],
    ) -> InferenceDecision:
        decision = await nxt()
        if isinstance(decision, ToolCallDecision):
            for call in decision.calls:
                self._record_and_check(call.name, call.arguments)
        return decision
