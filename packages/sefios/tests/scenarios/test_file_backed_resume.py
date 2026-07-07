"""File-backed pause/resume across simulated process restarts.

Exercises the real ``HumanInputTool`` with glyff's ``JsonFileBackend`` and the
sefios ``FileSessionStore``: every object (backend, store, tool, LLM client) is
constructed fresh for the second run, so the only thing connecting the two runs
is what was committed to disk before the pause. The resumed run must read back
the *same* interaction id the paused run stored — the idempotency hinge of the
human-in-the-loop flow.
"""

import json

import glyff
import pytest
from glyff_file_store import JsonFileBackend
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer
from sefia import Session, infer
from sefia.llm import LLMResponse

from sefios import FileSessionStore, NeedsInput
from sefios._session_state import bind_session_state
from sefios.tools import HumanInputRequest, HumanInputTool

_SESSION_ID = "file-backed-resume-test"

_TOOL_CALL_RESPONSE = LLMResponse(
    content=json.dumps(
        {
            "decision": "tool_calls",
            "tool_calls": [
                {
                    "name": "HumanInputTool_get_human_input",
                    "arguments": {"question": "What is your name?"},
                }
            ],
        }
    )
)
_RESULT_RESPONSE = LLMResponse(
    content=json.dumps({"decision": "result", "result": "The user's name is Alice."})
)


class _Agent:
    def __init__(self, tool: HumanInputTool):
        self._tool = tool

    @infer
    async def get_user_name(self) -> str:
        """Ask the user for their name and report it."""
        ...


async def test_pause_resume_survives_process_restart(tmp_path, make_mock_llm):
    seen: list[HumanInputRequest] = []
    answers: dict[str, str] = {}

    def get_answer(request: HumanInputRequest) -> str | None:
        seen.append(request)
        return answers.get(request.interaction_id)

    def make_glyff_session() -> glyff.Session:
        # A fresh backend instance per run, like a new process would create.
        return glyff.Session(
            id=_SESSION_ID,
            backend=JsonFileBackend(
                base_dir=tmp_path / "glyff", session_id=_SESSION_ID
            ),
            serializer=PydanticSerializer(),
            hasher=PydanticArgsHasher(),
        )

    def make_state_store() -> FileSessionStore:
        return FileSessionStore(
            base_dir=tmp_path / "state", serializer=PydanticSerializer()
        )

    # --- First run: no answer available, the run pauses. ---
    mock_llm = make_mock_llm([_TOOL_CALL_RESPONSE])
    with pytest.raises(NeedsInput):
        async with make_glyff_session() as gs:
            with bind_session_state(make_state_store()):
                async with Session(llm_client=mock_llm, glyff_session=gs):
                    await _Agent(HumanInputTool(get_answer=get_answer)).get_user_name()

    assert len(seen) == 1
    answers[seen[0].interaction_id] = "Alice"

    # --- Second run: everything is rebuilt from disk; the answer is found. ---
    resumed_llm = make_mock_llm([_RESULT_RESPONSE])
    async with make_glyff_session() as gs:
        with bind_session_state(make_state_store()):
            async with Session(llm_client=resumed_llm, glyff_session=gs):
                answer = await _Agent(
                    HumanInputTool(get_answer=get_answer)
                ).get_user_name()

    assert answer == "The user's name is Alice."
    # The resumed call read the same interaction id back from the file store,
    # so the pending question was keyed stably across the restart.
    assert [r.interaction_id for r in seen] == [seen[0].interaction_id] * 2
    # The completed first step was replayed from the engraved record: the
    # resumed run only asked the LLM for the final decision.
    assert len(resumed_llm.requests) == 1
