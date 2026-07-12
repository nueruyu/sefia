"""Durable history + compaction across a simulated process restart.

The distinguishing property of ``DurableHistoryStore``: the run's history is
stored state, independent of glyff's engraved execution log. A
``HistoryCompactor`` can therefore rewrite it mid-run without breaking replay,
and a resumed run continues from the compacted snapshot — the compacted-away
steps are never replayed, and the model never sees them again.

Every object (glyff backend, session storage, tools, LLM client) is rebuilt
for the second run, so the only bridge between the runs is what was committed
to disk before the pause.
"""

import json

import glyff
import pytest
from glyff_file_store import JsonFileBackend
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer
from sefia import Policy, Session, infer
from sefia.llm import LLMResponse

from sefios import DurableHistoryStore, FileSessionStorage, NeedsInput
from sefios._session_state import bind_session_storage
from sefios.middleware import HistoryCompactor
from sefios.tools import InputRequest, InputTool

_SESSION_ID = "history-compaction-test"


def _note_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=json.dumps(
            {
                "decision": "tool_calls",
                "tool_calls": [{"name": "Notes_add_note", "arguments": {"text": text}}],
            }
        )
    )


_ASK_RESPONSE = LLMResponse(
    content=json.dumps(
        {
            "decision": "tool_calls",
            "tool_calls": [
                {
                    "name": "InputTool_get_input",
                    "arguments": {"prompt": "Anything else?"},
                }
            ],
        }
    )
)
_RESULT_RESPONSE = LLMResponse(
    content=json.dumps({"decision": "result", "result": "All done."})
)


class Notes:
    async def add_note(self, text: str) -> str:
        return f"noted: {text}"


class _Agent:
    def __init__(self, notes: Notes, input_tool: InputTool):
        self._notes = notes
        self._input = input_tool

    @infer
    async def chat(self) -> str:
        """Take notes for the user and confirm before finishing."""
        ...


async def test_compacted_history_survives_restart_without_replaying_old_steps(
    tmp_path, make_mock_llm
):
    seen: list[InputRequest] = []
    answers: dict[str, str] = {}

    def get_input(request: InputRequest) -> str | None:
        seen.append(request)
        return answers.get(request.interaction_id)

    def make_glyff_session() -> glyff.Session:
        return glyff.Session(
            id=_SESSION_ID,
            backend=JsonFileBackend(
                base_dir=tmp_path / "glyff", session_id=_SESSION_ID
            ),
            serializer=PydanticSerializer(),
            hasher=PydanticArgsHasher(),
        )

    def make_state_storage() -> FileSessionStorage:
        return FileSessionStorage(
            base_dir=tmp_path / "state", serializer=PydanticSerializer()
        )

    # Compacts once the history exceeds 5 items, keeping the last completed
    # step: after three note-taking steps (6 items) only the third survives.
    compaction_policy = Policy(
        middleware=lambda: [HistoryCompactor(max_items=5, keep_items=2)]
    )

    # --- First run: three tool steps, compaction, then a pause. ---
    mock_llm = make_mock_llm(
        [
            _note_response("zero"),
            _note_response("one"),
            _note_response("two"),
            _ASK_RESPONSE,
        ]
    )
    with pytest.raises(NeedsInput):
        async with make_glyff_session() as gs:
            with bind_session_storage(make_state_storage()):
                async with Session(
                    llm_client=mock_llm,
                    glyff_session=gs,
                    policies=[compaction_policy],
                    history_store=DurableHistoryStore(),
                ):
                    await _Agent(Notes(), InputTool(get_input=get_input)).chat()

    # The fourth model call ran after compaction: the model saw fewer history
    # messages than the (uncompacted) third call did.
    assert len(mock_llm.requests) == 4
    assert len(mock_llm.requests[3]["messages"]) < len(mock_llm.requests[2]["messages"])

    answers[seen[0].interaction_id] = "No, that's all."

    # --- Second run: rebuilt from disk; continues from the compacted history. ---
    resumed_llm = make_mock_llm([_RESULT_RESPONSE])
    async with make_glyff_session() as gs:
        with bind_session_storage(make_state_storage()):
            async with Session(
                llm_client=resumed_llm,
                glyff_session=gs,
                policies=[compaction_policy],
                history_store=DurableHistoryStore(),
            ):
                result = await _Agent(Notes(), InputTool(get_input=get_input)).chat()

    assert result == "All done."
    # One model call: the paused step replayed from its engraved record (keyed
    # on the compacted history the store handed back), and the compacted-away
    # steps were never re-entered at all.
    assert len(resumed_llm.requests) == 1
    # The model still sees the surviving step but not the compacted-away ones.
    resumed_messages = json.dumps(resumed_llm.requests[0]["messages"])
    assert "noted: two" in resumed_messages
    assert "noted: zero" not in resumed_messages
