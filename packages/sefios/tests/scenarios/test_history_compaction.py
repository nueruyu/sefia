"""Durable history + compaction across a simulated process restart.

Runs against both durable backends: the default ``GlyffHistoryStorage`` (history
in the run's glyff metadata) and ``SessionHistoryStorage`` (history in the
session storage). Every object is rebuilt for the second run, so the only bridge
between runs is what was committed to disk before the pause.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from sefia import HistoryStorage, Policy, Tools
from sefia.inference import ToolCallResult
from sefia.llm import DecisionPrompt, LLMCompletion, PromptRenderer
from sefia.testing import MockLLMClient, result_completion, tool_calls_completion

from sefios import SessionScope, SQLitePersistence, domain
from sefios.exceptions import InputRequired
from sefios.history_storages import SessionHistoryStorage
from sefios.middleware import HistoryCompactor
from sefios.tools import Input, InputRequest

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
        self.prompts: list[DecisionPrompt] = []

    def render(self, prompt: DecisionPrompt) -> str:
        self.prompts.append(prompt)
        return "prompt"

    def render_tool_result(self, result: ToolCallResult) -> str:
        return str(result.result)


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
    seen: list[InputRequest] = []
    answers: dict[str, str] = {}

    def get_input(request: InputRequest) -> str | None:
        seen.append(request)
        return answers.get(request.interaction_id)

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
        middleware=lambda: [HistoryCompactor(max_items=5, keep_items=2)]
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
    with pytest.raises(InputRequired):
        async with make_scope(mock_llm).session(session_id=_SESSION_ID):
            await _Agent(Notes(), Input(get_input=get_input)).chat()

    assert [len(prompt.history) for prompt in renderer.prompts] == [0, 2, 4, 2]

    answers[seen[0].interaction_id] = "No, that's all."

    resumed_llm = make_mock_llm([_RESULT_RESPONSE])
    async with make_scope(resumed_llm).session(session_id=_SESSION_ID):
        result = await _Agent(Notes(), Input(get_input=get_input)).chat()

    assert result == "All done."
    assert len(resumed_llm.requests) == 1
    resumed_results = [
        item.result
        for item in renderer.prompts[-1].history
        if isinstance(item, ToolCallResult)
    ]
    assert "noted: two" in resumed_results
    assert "noted: zero" not in resumed_results
