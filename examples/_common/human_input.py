from dataclasses import asdict

from rich.console import Console
from sefia import get_context
from sefios.tools import HumanInputRequest, HumanInputResult, HumanInputTool

_PENDING_HUMAN_INTERACTION_KEY = "pending_human_interaction"


class ChatHumanInputAdapter:
    """Connects HumanInputTool callbacks to the example chat session protocol."""

    def __init__(self, console: Console):
        self._console = console

    def create_tool(self) -> HumanInputTool:
        return HumanInputTool(
            get_answer=self.get_answer,
            on_request=self.on_request,
            on_complete=self.on_complete,
        )

    async def get_answer(self, request: HumanInputRequest) -> str | None:
        session_store = get_context().session_store
        return await session_store.get(self._answer_key(request.interaction_id), str)

    async def on_request(self, request: HumanInputRequest) -> None:
        session_store = get_context().session_store
        await session_store.set(_PENDING_HUMAN_INTERACTION_KEY, asdict(request), dict)
        self._console.print(f"\n[bold yellow][USER_INPUT_REQUIRED][/bold yellow] {request.question}\n")

    async def on_complete(self, result: HumanInputResult) -> None:
        session_store = get_context().session_store
        await session_store.delete(self._answer_key(result.interaction_id))
        await session_store.delete(_PENDING_HUMAN_INTERACTION_KEY)

    @staticmethod
    def _answer_key(interaction_id: str) -> str:
        return f"human_input__{interaction_id}"
