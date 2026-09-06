from collections.abc import Awaitable, Callable
from typing import final

from sefia.event_system import EventHandler
from sefia.events import ToolExecutionBound
from typing_extensions import override

from .._input import preview_id_for
from .._session_state import execution_id_scope_key


@final
class InputBindingPublisher(EventHandler[ToolExecutionBound]):
    """Translate execution observations into public input identifiers."""

    def __init__(self, publish: Callable[[str, str], Awaitable[None]]) -> None:
        self._publish = publish

    @override
    async def handle(self, event: ToolExecutionBound) -> None:
        execution = event.execution_id
        if (
            execution.domain_id.value != "sefios"
            or execution.name.value != "tools.input.get_input"
        ):
            return
        await self._publish(
            preview_id_for(event.tool_call_id), execution_id_scope_key(execution)
        )
