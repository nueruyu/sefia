"""Durable history + compaction across a simulated process restart.

Runs against both durable backends: the default ``GlyffHistoryStorage`` (history
in the run's glyff metadata) and ``SessionHistoryStorage`` (history in the
session storage). Every object is rebuilt for the second run, so the only bridge
between runs is what was committed to disk before the pause.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sefia import HistoryStorage, MiddlewareSet, Policy, Tools
from sefia.llm import InferencePrompt, LLMCompletion, PromptRenderer
from sefia.testing import MockLLMClient, result_completion, tool_calls_completion
from sefios import SessionScope, SQLitePersistence, domain
from sefios.exceptions import InteractionRequired
from sefios.history_storages import SessionHistoryStorage
from sefios.interactions import InteractionChannel
from sefios.middleware import HistoryCompactor
from sefios.tools import Input
from typing_extensions import override

infer = domain(
    "packages.sefios.tests.scenarios.test_history_compaction", version="1"
).infer

_SESSION_ID = "history-compaction-test"


def _note_response(text: str) -> LLMCompletion:
    return tool_calls_completion(("Notes_add_note", {"text": text}))


_ASK_RESPONSE = tool_calls_completion(("Input_get_input", {"prompt": "Anything else?"}))
_RESULT_RESPONSE = result_completion("All done.")


def _session_history_storage() -> HistoryStorage:
    return SessionHistoryStorage()


class Notes:
    async def add_note(self, text: str) -> str:
        return f"noted: {text}"


class _RecordingRenderer(PromptRenderer):
    def __init__(self) -> None:
        self.prompts: list[InferencePrompt] = []

    @override
    def render(self, prompt: InferencePrompt) -> str:
        self.prompts.append(prompt)
        return "prompt"


def _history_records(request: dict[str, Any]) -> list[dict[str, Any]]:
    text = next(
        message["content"]
        for message in request["messages"]
        if message["content"].startswith("## Previous tool interactions")
    )
    return json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])


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
    make_mock_llm: Callable[[list[LLMCompletion]], MockLLMClient],
    make_history_storage: Callable[[], HistoryStorage] | None,
) -> None:
    def make_scope(client: MockLLMClient) -> SessionScope:
        return SessionScope(
            llm_client=client,
            persistence=SQLitePersistence(tmp_path / "sessions.sqlite3"),
            policies=[compaction_policy],
            history_storage=(
                make_history_storage() if make_history_storage is not None else None
            ),
            prompt_renderer=renderer,
        )

    compaction_policy = Policy(
        middleware=lambda: MiddlewareSet(
            step=(HistoryCompactor(max_items=5, keep_items=2),)
        )
    )
    renderer = _RecordingRenderer()

    mock_llm = make_mock_llm(
        [
            _note_response("zero"),
            _note_response("one"),
            _note_response("two"),
            _ASK_RESPONSE,
        ]
    )
    with pytest.raises(InteractionRequired) as pause_info:
        async with make_scope(mock_llm).session(session_id=_SESSION_ID):
            await _Agent(Notes(), Input()).chat()

    assert [len(request["messages"]) for request in mock_llm.requests] == [1, 3, 3, 3]
    assert [len(_history_records(request)) for request in mock_llm.requests[1:]] == [
        2,
        4,
        2,
    ]

    channel = InteractionChannel(
        SQLitePersistence(tmp_path / "sessions.sqlite3").create_session_storage(
            _SESSION_ID
        )
    )
    resumed_llm = make_mock_llm([_RESULT_RESPONSE])
    async with make_scope(resumed_llm).session(session_id=_SESSION_ID):
        await channel.resolve(pause_info.value.interaction_id, "Alice")
        result = await _Agent(Notes(), Input()).chat()

    assert result == "All done."
    assert len(resumed_llm.requests) == 1
    resumed_results = [
        record["tool_result"]["result"]
        for record in _history_records(resumed_llm.requests[-1])
        if "tool_result" in record
    ]
    assert "noted: two" in resumed_results
    assert "noted: zero" not in resumed_results
