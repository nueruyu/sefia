import json
from collections import deque
from typing import Any, Awaitable, Callable

from ..interfaces.middleware import StepContext, StepMiddleware
from ..models import InferenceDecision, ToolCallDecision
from .signals import StagnationError


class StagnationMiddleware(StepMiddleware):
    """
    Detects if the agent is stagnating by repeating the same tool call.

    The middleware inspects each step's decision and records its tool calls. If
    the same call recurs ``max_repeats`` times in a row it raises
    ``StagnationError`` before the repeated tool runs again.

    The rolling history is intentionally kept on the instance. Middleware is
    instantiated per inference run (``Policy.create_middleware`` is called once
    per ``@infer`` invocation in ``decorators._run``), so an instance is never
    shared across concurrent runs; its state is scoped to a single run. The
    history is reset at ``ctx.step == 0`` so a retried attempt starts clean.
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

    async def wrap(
        self,
        ctx: StepContext,
        nxt: Callable[[], Awaitable[InferenceDecision]],
    ) -> InferenceDecision:
        # A new attempt starts the step count over at 0. The same middleware
        # instance is reused across retries, so reset the rolling history here to
        # avoid carrying tool calls from a previous (failed) attempt forward,
        # which would otherwise trip a false-positive StagnationError.
        if ctx.step == 0:
            self.history.clear()
        decision = await nxt()
        if isinstance(decision, ToolCallDecision):
            for call in decision.calls:
                self._record_and_check(call.name, call.arguments)
        return decision
