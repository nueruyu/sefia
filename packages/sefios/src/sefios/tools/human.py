import uuid
from dataclasses import dataclass

from glyff import engrave
from glyff.exceptions import YieldException
from sefia import get_context, tool


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
        ctx = get_context()
        call_store = ctx.get_call_state_store("internal_state", _AskUserState)
        call_state = await call_store.ensure()
        session_store = ctx.session_store

        if call_state.interaction_id is None:
            # First call: generate an ID, save it to call-local state, and interrupt.
            interaction_id = str(uuid.uuid4())
            call_state.interaction_id = interaction_id
            await call_store.save(call_state)

            # Signal to the runner that we are waiting for input for this interaction.
            interaction_details = {"id": interaction_id, "question": question}
            await session_store.set(
                "pending_human_interaction", interaction_details, dict
            )
            self._prompt_user_input(question)
            raise YieldException()

        # Resumed call: check for the answer provided by the runner.
        interaction_id = call_state.interaction_id
        answer_key = f"human_input__{interaction_id}"
        answer = await session_store.get(answer_key, str)

        if answer is not None:
            await session_store.delete(answer_key)
            await session_store.delete("pending_human_interaction")
            return answer

        # No answer provided yet, prompt again and re-yield.
        self._prompt_user_input(question)
        raise YieldException()
