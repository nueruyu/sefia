"""File-backed pause/resume across simulated process restarts.

Exercises the real ``Input`` with glyff and Sefia state sharing SQLite: every
object (backend, store, tool, LLM client) is
constructed fresh for the second run, so the only thing connecting the two runs
is what was committed to disk before the pause. The resumed run must read back
the *same* interaction id the paused run stored — the idempotency hinge of the
human-in-the-loop flow.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from sefia import Tools
from sefia.llm import LLMCompletion
from sefia.testing import MockLLMClient, result_completion, tool_calls_completion
from sefios import SessionScope, SQLitePersistence, domain
from sefios.exceptions import InteractionRequired
from sefios.interactions import InteractionChannel, InteractionRequest
from sefios.tools import Input

infer = domain(
    "packages.sefios.tests.scenarios.test_file_backed_resume", version="1"
).infer

_SESSION_ID = "file-backed-resume-test"

_TOOL_CALL_RESPONSE = tool_calls_completion(
    ("Input_get_input", {"prompt": "What is your name?"})
)
_RESULT_RESPONSE = result_completion("The user's name is Alice.")


class _Agent:
    _tool: Tools[Input]

    def __init__(self, tool: Input):
        self._tool = tool

    @infer
    async def get_user_name(self) -> str:
        """Ask the user for their name and report it."""
        ...


async def test_pause_resume_survives_process_restart(
    tmp_path: Path,
    make_mock_llm: Callable[[list[LLMCompletion]], MockLLMClient],
) -> None:
    def make_scope(client: MockLLMClient) -> SessionScope:
        return SessionScope(
            llm_client=client,
            persistence=SQLitePersistence(tmp_path / "sessions.sqlite3"),
        )

    mock_llm = make_mock_llm([_TOOL_CALL_RESPONSE])
    with pytest.raises(InteractionRequired) as pause_info:
        async with make_scope(mock_llm).session(session_id=_SESSION_ID):
            await _Agent(Input()).get_user_name()

    channel = InteractionChannel(
        SQLitePersistence(tmp_path / "sessions.sqlite3").create_session_storage(
            _SESSION_ID
        )
    )
    assert await channel.pending() == [
        InteractionRequest(
            pause_info.value.interaction_id,
            {"type": "input", "prompt": "What is your name?"},
        )
    ]

    resumed_llm = make_mock_llm([_RESULT_RESPONSE])
    async with make_scope(resumed_llm).session(session_id=_SESSION_ID):
        await channel.resolve(pause_info.value.interaction_id, "Alice")
        answer = await _Agent(Input()).get_user_name()

    assert answer == "The user's name is Alice."
    assert await channel.pending() == []
    # The completed first step was replayed from the engraved record: the
    # resumed run only asked the LLM for the final decision.
    assert len(resumed_llm.requests) == 1
