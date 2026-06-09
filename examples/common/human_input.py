import uuid
from dataclasses import dataclass

import sefia.context
from glyff import engrave
from glyff.exceptions import YieldException
from sefia import tool

from .chat_session import ChatSessionState


@dataclass
class _AskUserState:
    """Internal state for a single ask_user tool call."""

    interaction_id: str | None = None


@dataclass
class HumanInputTool:
    @staticmethod
    def _prompt_user_input(question: str) -> None:
        print(f"\n[USER_INPUT_REQUIRED] {question}\n")

    @tool
    @engrave
    async def get_human_input(self, question: str) -> str:
        """
        Asks the user a question and returns their answer.
        This tool interrupts the session to wait for user input.
        """
        ctx = sefia.context.get_context()
        call_store = ctx.get_call_state_store("internal_state", _AskUserState)
        call_state = await call_store.ensure()

        session_store = ctx.get_state_store("session_state", ChatSessionState)
        session_state = await session_store.ensure()

        if call_state.interaction_id is None:
            # First call: generate an ID, save it, and interrupt.
            interaction_id = str(uuid.uuid4())
            call_state.interaction_id = interaction_id
            session_state.add_pending_interaction(interaction_id)
            await call_store.save(call_state)
            await session_store.save(session_state)
            self._prompt_user_input(question)
            raise YieldException()

        # Resumed call: check for the answer.
        interaction_id = call_state.interaction_id
        answer = session_state.get_answer_by_id(interaction_id)
        if answer is not None:
            return answer

        # No answer provided yet, interrupt again.
        self._prompt_user_input(question)
        raise YieldException()
