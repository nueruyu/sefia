"""Durable history + compaction across a simulated process restart.

Runs against both durable backends: the default ``GlyffHistoryStorage`` (history
in the run's glyff metadata) and ``SessionHistoryStorage`` (history in the
session storage). Every object is rebuilt for the second run, so the only bridge
between runs is what was committed to disk before the pause.
"""

import json
from collections.abc import Callable
from pathlib import Path

import glyff
from glyff.serialization import FallbackByTypeQualname
import pytest
from glyff_pydantic import PydanticArgumentCanonicalizer, PydanticSerializer
from glyff_sqlite import SQLiteBackend
from sefia import HistoryStorage, Policy, Session, Tools
from sefia.llm import LLMResponse
from sefia.testing import MockLLMClient, result_response, tool_calls_response

from sefios import SQLiteSessionStorage, domain
from sefios.exceptions import InputRequired
from sefios.history_storages import SessionHistoryStorage
from sefios._session_state import bind_session_storage
from sefios.middleware import HistoryCompactor
from sefios.tools import Input, InputRequest

infer = domain(
    "packages.sefios.tests.scenarios.test_history_compaction", version="1"
).infer

_SESSION_ID = "history-compaction-test"


def _note_response(text: str) -> LLMResponse:
    return tool_calls_response(("Notes_add_note", {"text": text}))


_ASK_RESPONSE = tool_calls_response(("Input_get_input", {"prompt": "Anything else?"}))
_RESULT_RESPONSE = result_response("All done.")


def _session_history_storage() -> HistoryStorage:
    return SessionHistoryStorage()


class Notes:
    async def add_note(self, text: str) -> str:
        return f"noted: {text}"


class _Agent:
    _notes: Tools[Notes]
    _input: Tools[Input]

    def __init__(self, notes: Notes, input_tool: Input):
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
    [None, _session_history_storage],
    ids=["glyff-metadata", "session-storage"],
)
async def test_compacted_history_survives_restart_without_replaying_old_steps(
    tmp_path: Path,
    make_mock_llm: Callable[[list[LLMResponse]], MockLLMClient],
    make_history_storage: Callable[[], HistoryStorage] | None,
) -> None:
    seen: list[InputRequest] = []
    answers: dict[str, str] = {}

    def get_input(request: InputRequest) -> str | None:
        seen.append(request)
        return answers.get(request.interaction_id)

    def make_glyff_session() -> glyff.Session:
        return glyff.Session(
            id=glyff.SessionId(_SESSION_ID),
            backend=SQLiteBackend(tmp_path / "sessions.sqlite3"),
            serializer=PydanticSerializer(),
            argument_canonicalizer=PydanticArgumentCanonicalizer(
                FallbackByTypeQualname()
            ),
        )

    def make_state_storage() -> SQLiteSessionStorage:
        return SQLiteSessionStorage(
            tmp_path / "sessions.sqlite3", _SESSION_ID, PydanticSerializer()
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
    with pytest.raises(InputRequired):
        async with make_glyff_session() as gs:
            with bind_session_storage(make_state_storage()):
                async with Session(
                    llm_client=mock_llm,
                    glyff_session=gs,
                    policies=[compaction_policy],
                    history_storage=history_storage,
                ):
                    await _Agent(Notes(), Input(get_input=get_input)).chat()

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
                result = await _Agent(Notes(), Input(get_input=get_input)).chat()

    assert result == "All done."
    assert len(resumed_llm.requests) == 1
    resumed_messages = json.dumps(resumed_llm.requests[0]["messages"])
    assert "noted: two" in resumed_messages
    assert "noted: zero" not in resumed_messages
