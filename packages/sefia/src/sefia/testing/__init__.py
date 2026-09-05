"""Test helpers and conformance contracts for building against sefia.

This is a public, supported surface: applications embedding sefia can use it
in their own test suites, and the workspace packages' tests use it too. It
gives shared helpers a collision-free import path (``sefia.testing``) so test
trees themselves can stay plain, non-importable directories.

Application tests can use the scripted client and in-memory session::

    from sefia.testing import MockLLMClient, memory_session, result_completion

    async def test_answer():
        llm = MockLLMClient(completions=[result_completion("hi")])
        async with memory_session(llm):
            assert await my_agent.answer(question="greet") == "hi"

Extension authors can install ``sefia[testing]`` and subclass the exported
conformance contracts in their own pytest suites.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import fields, is_dataclass, replace
from typing import Any, AsyncGenerator, Callable, Coroutine, cast

import glyff
from glyff.serialization import (
    FallbackByTypeQualname,
)
from glyff.store import MemoryBackend
from glyff_pydantic import (
    PydanticArgumentCanonicalizer,
    PydanticSerializer,
)
from typing_extensions import final, override

from .._interfaces.history_storage import HistorySnapshot, HistoryStorage
from .._session import Session
from ..llm import LLMClient, LLMCompletion, Message
from ..llm.step_decision import DecisionSpec, StepTool
from ..llm.structured_data import StructuredData
from ..llm.streaming import OutputStreamCallback, OutputStreamEvent
from ..llm.transports import DecisionObserver
from ..pydantic._json_utils import pydantic_json_default
from ._decision_transport_contract import (
    DecisionTransportCase,
    DecisionTransportContract,
)
from ._history_storage_contract import HistoryStorageContract
from ._tool_collector_contract import ToolCollectorCase, ToolCollectorContract


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, StructuredData):
        return _snapshot_value(value.tree)
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, Any] = {}
        for item in fields(value):
            converted = _snapshot_value(getattr(value, item.name))
            if converted is not None:
                result[item.name] = converted
        return result
    if isinstance(value, dict):
        return {
            key: _snapshot_value(item)
            for key, item in cast(dict[str, Any], value).items()
        }
    if isinstance(value, list):
        return [_snapshot_value(item) for item in cast(list[Any], value)]
    return value


@final
class MockLLMClient(LLMClient):
    """An ``LLMClient`` that replays scripted ``completions`` and records every
    request it receives in ``requests`` (messages as plain dicts, plus the
    tools, output schema, and callbacks)."""

    def __init__(self, completions: list[LLMCompletion]):
        self.completions = list(completions)
        self.requests: list[dict[str, Any]] = []

    @override
    async def complete(
        self,
        messages: list[Message],
        tools: list[StepTool] | None = None,
        decision_spec: DecisionSpec | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
        output_callback: OutputStreamCallback | None = None,
        reasoning_callback: (
            Callable[[str], Coroutine[None, None, None]] | None
        ) = None,
    ) -> LLMCompletion:
        self.requests.append(
            {
                "messages": [_snapshot_value(message) for message in messages],
                "tools": tools,
                "decision_spec": decision_spec,
                "stream_callback": stream_callback,
                "output_callback": output_callback,
                "reasoning_callback": reasoning_callback,
            }
        )
        if not self.completions:
            raise AssertionError("MockLLMClient has no more completions.")
        completion = self.completions.pop(0)
        if (
            decision_spec is not None
            and completion.structured_output is None
            and completion.content is not None
        ):
            try:
                completion = replace(
                    completion,
                    structured_output=StructuredData.parse_json(completion.content),
                )
            except json.JSONDecodeError:
                pass
        return completion


@final
class MemoryHistoryStorage(HistoryStorage):
    """In-memory ``HistoryStorage``; records every saved snapshot in ``saves``.

    Each entry is its own ``HistorySnapshot`` (the items tuple is copied on
    save), so later saves cannot rewrite earlier records. The history items
    themselves are shared by reference.
    """

    def __init__(self, initial: HistorySnapshot | None = None):
        self.snapshot = initial if initial is not None else HistorySnapshot()
        self.saves: list[HistorySnapshot] = []

    @override
    async def load(self) -> HistorySnapshot:
        return self.snapshot

    @override
    async def save(self, snapshot: HistorySnapshot) -> None:
        record = HistorySnapshot(
            items=tuple(snapshot.items), completed_steps=snapshot.completed_steps
        )
        self.snapshot = record
        self.saves.append(record)


class RecordingDecisionObserver(DecisionObserver):
    """Records decision transport callbacks for assertions in tests."""

    def __init__(self) -> None:
        self.prompt: str | None = None
        self.prompts: list[str] = []
        self.response_texts: list[str] = []
        self.reasoning_texts: list[str] = []
        self.output_events: list[OutputStreamEvent] = []

    @override
    async def before_request(self, prompt: str) -> None:
        self.prompt = prompt
        self.prompts.append(prompt)

    @override
    async def response_text(self, text: str) -> None:
        self.response_texts.append(text)

    @override
    async def reasoning_text(self, text: str) -> None:
        self.reasoning_texts.append(text)

    @override
    async def output(self, event: OutputStreamEvent) -> None:
        self.output_events.append(event)


def result_completion(result: Any) -> LLMCompletion:
    """A scripted "result" decision carrying ``result`` as the final answer.

    ``result`` may be anything the framework's JSON encoding handles —
    including dataclasses and Pydantic models, which serialize to the object
    shape the step-decision schema validates.
    """
    return LLMCompletion(
        content=json.dumps(
            {"decision": "result", "result": result},
            default=pydantic_json_default,
        )
    )


def tool_calls_completion(*calls: tuple[str, dict[str, Any]]) -> LLMCompletion:
    """A scripted "tool_calls" decision from ``(tool_name, arguments)`` pairs."""
    return LLMCompletion(
        content=json.dumps(
            {
                "decision": "tool_calls",
                "tool_calls": [
                    {"name": name, "arguments": arguments} for name, arguments in calls
                ],
            },
            default=pydantic_json_default,
        )
    )


@asynccontextmanager
async def memory_session(
    llm_client: LLMClient,
    *,
    session_id: str = "test-session",
    backend: Any | None = None,
    **session_kwargs: Any,
) -> AsyncGenerator[Session]:
    """A ready-to-use sefia ``Session`` over an in-memory glyff backend.

    Pass a shared ``backend`` with a stable ``session_id`` to simulate
    pause/resume across runs. Extra keyword arguments go to ``Session``.
    """
    async with glyff.Session(
        id=glyff.SessionId(session_id),
        backend=backend if backend is not None else MemoryBackend(),
        serializer=PydanticSerializer(),
        argument_canonicalizer=PydanticArgumentCanonicalizer(FallbackByTypeQualname()),
    ) as glyff_session:
        async with Session(
            llm_client=llm_client, glyff_session=glyff_session, **session_kwargs
        ) as session:
            yield session


__all__ = [
    "DecisionTransportCase",
    "DecisionTransportContract",
    "HistoryStorageContract",
    "MemoryHistoryStorage",
    "MockLLMClient",
    "RecordingDecisionObserver",
    "ToolCollectorCase",
    "ToolCollectorContract",
    "memory_session",
    "result_completion",
    "tool_calls_completion",
]
