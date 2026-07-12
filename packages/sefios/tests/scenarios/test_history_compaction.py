"""Durable history + compaction across a simulated process restart.

Runs against both durable backends: the default ``GlyffHistoryStorage`` (history
in the run's glyff metadata) and ``SessionHistoryStorage`` (history in the
session storage). Every object is rebuilt for the second run, so the only bridge
between runs is what was committed to disk before the pause.
"""

import json

import glyff
import pytest
from glyff_file_store import JsonFileBackend
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer
from sefia import Policy, Session, infer
from sefia.llm import LLMResponse

from sefios import FileSessionStorage, NeedsInput, SessionHistoryStorage
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


# None exercises the default GlyffHistoryStorage (glyff metadata); the factory
# exercises the sefios SessionStorage-backed alternative.
@pytest.mark.parametrize(
    "make_history_storage",
    [None, lambda: SessionHistoryStorage()],
    ids=["glyff-metadata", "session-storage"],
)
async def test_compacted_history_survives_restart_without_replaying_old_steps(
    tmp_path, make_mock_llm, make_history_storage
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

    history_storage = make_history_storage() if make_history_storage else None
    compaction_policy = Policy(
        middleware=lambda: [HistoryCompactor(max_items=5, keep_items=2)]
    )

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
                    history_storage=history_storage,
                ):
                    await _Agent(Notes(), InputTool(get_input=get_input)).chat()

    assert len(mock_llm.requests) == 4
    assert len(mock_llm.requests[3]["messages"]) < len(mock_llm.requests[2]["messages"])

    answers[seen[0].interaction_id] = "No, that's all."

    resumed_llm = make_mock_llm([_RESULT_RESPONSE])
    async with make_glyff_session() as gs:
        with bind_session_storage(make_state_storage()):
            async with Session(
                llm_client=resumed_llm,
                glyff_session=gs,
                policies=[compaction_policy],
                history_storage=history_storage,
            ):
                result = await _Agent(Notes(), InputTool(get_input=get_input)).chat()

    assert result == "All done."
    assert len(resumed_llm.requests) == 1
    resumed_messages = json.dumps(resumed_llm.requests[0]["messages"])
    assert "noted: two" in resumed_messages
    assert "noted: zero" not in resumed_messages
