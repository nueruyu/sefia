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
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
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
from ..pydantic import PydanticStructuredDataConverter
from ._decision_transport_contract import (
    DecisionTransportCase,
    DecisionTransportContract,
)
from ._factories import (
    make_decision_context,
    make_decision_request,
    make_function_info,
    make_step_context,
    make_tool_call_request,
)
from ._history_storage_contract import HistoryStorageContract
from ._llm_client_contract import (
    LLMClientCase,
    LLMClientContract,
    StreamingLLMClientCase,
    StreamingLLMClientContract,
)
from ._tool_collector_contract import ToolCollectorCase, ToolCollectorContract


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, StructuredData):
        return _snapshot_value(value.tree)
    if isinstance(value, Message):
        result: dict[str, Any] = {"role": value.role}
        content = value.content
        if content is not None:
            result["content"] = _snapshot_value(content)
        if value.tool_call_id is not None:
            result["tool_call_id"] = value.tool_call_id
        if value.tool_calls is not None:
            result["tool_calls"] = [_snapshot_value(call) for call in value.tool_calls]
        return result
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
@dataclass(frozen=True)
class ScriptedCompletion:
    """A final completion plus deterministic callback values for ``MockLLMClient``.

    Callback sequences are emitted only when the corresponding callback is supplied.
    The final ``completion`` remains the authoritative response returned by the client.
    """

    completion: LLMCompletion
    content_chunks: Sequence[str] = ()
    reasoning_chunks: Sequence[str] = ()
    output_events: Sequence[OutputStreamEvent] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_chunks", tuple(self.content_chunks))
        object.__setattr__(self, "reasoning_chunks", tuple(self.reasoning_chunks))
        object.__setattr__(self, "output_events", tuple(self.output_events))


@final
class MockLLMClient(LLMClient):
    """An ``LLMClient`` that replays scripted ``completions`` and records every
    request it receives in ``requests`` (messages as plain dicts, plus the
    tools, output schema, and callbacks)."""

    def __init__(
        self, completions: Sequence[LLMCompletion | ScriptedCompletion]
    ) -> None:
        self.completions: list[LLMCompletion | ScriptedCompletion] = list(completions)
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
        scripted = self.completions.pop(0)
        if isinstance(scripted, ScriptedCompletion):
            completion = scripted.completion
            if reasoning_callback is not None:
                for chunk in scripted.reasoning_chunks:
                    await reasoning_callback(chunk)
            if stream_callback is not None:
                for chunk in scripted.content_chunks:
                    await stream_callback(chunk)
            if output_callback is not None:
                for event in scripted.output_events:
                    await output_callback(event)
        else:
            completion = scripted

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

    Each entry is its own ``HistorySnapshot``, so replacing the current
    snapshot cannot rewrite earlier records. History items are shared by
    reference.
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
        self.messages: tuple[Message, ...] | None = None
        self.requests: list[tuple[Message, ...]] = []
        self.response_texts: list[str] = []
        self.reasoning_texts: list[str] = []
        self.output_events: list[OutputStreamEvent] = []

    @override
    async def before_request(self, messages: tuple[Message, ...]) -> None:
        self.messages = messages
        self.requests.append(messages)

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
    data = PydanticStructuredDataConverter().to_structured_data(
        {"decision": "result", "result": result}
    )
    return LLMCompletion(content=json.dumps(data.to_json_value()))


def tool_calls_completion(*calls: tuple[str, dict[str, Any]]) -> LLMCompletion:
    """A scripted "tool_calls" decision from ``(tool_name, arguments)`` pairs."""
    data = PydanticStructuredDataConverter().to_structured_data(
        {
            "decision": "tool_calls",
            "tool_calls": [
                {"name": name, "arguments": arguments} for name, arguments in calls
            ],
        }
    )
    return LLMCompletion(content=json.dumps(data.to_json_value()))


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
    "LLMClientCase",
    "LLMClientContract",
    "MemoryHistoryStorage",
    "MockLLMClient",
    "RecordingDecisionObserver",
    "ScriptedCompletion",
    "StreamingLLMClientCase",
    "StreamingLLMClientContract",
    "ToolCollectorCase",
    "ToolCollectorContract",
    "make_decision_context",
    "make_decision_request",
    "make_function_info",
    "make_step_context",
    "make_tool_call_request",
    "memory_session",
    "result_completion",
    "tool_calls_completion",
]
