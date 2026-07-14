"""Test doubles and helpers for writing tests against sefia.

This is a public, supported surface: applications embedding sefia can use it
in their own test suites, and the workspace packages' tests use it too. It
gives shared helpers a collision-free import path (``sefia.testing``) so test
trees themselves can stay plain, non-importable directories.

Typical shape of a test::

    from sefia.testing import MockLLMClient, memory_session, result_response

    async def test_answer():
        llm = MockLLMClient(responses=[result_response("hi")])
        async with memory_session(llm):
            assert await my_agent.answer(question="greet") == "hi"
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Coroutine

import glyff
from glyff.store import MemoryBackend
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer

from ._interfaces.history_storage import HistorySnapshot, HistoryStorage
from ._session import Session
from .llm import LLMClient, LLMResponse, Message
from .pydantic._json_utils import pydantic_json_default


class MockLLMClient(LLMClient):
    """An ``LLMClient`` that replays scripted responses and records requests.

    Each ``complete()`` call pops the next response from ``responses`` and
    appends the request it received to ``requests`` (messages as plain dicts,
    plus the tools, output schema, and callbacks), so tests can assert on
    exactly what reached the model.
    """

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        output_schema: dict | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
        reasoning_callback: (
            Callable[[str], Coroutine[None, None, None]] | None
        ) = None,
    ) -> LLMResponse:
        self.requests.append(
            {
                "messages": [m.to_dict(exclude_none=True) for m in messages],
                "tools": tools,
                "output_schema": output_schema,
                "stream_callback": stream_callback,
                "reasoning_callback": reasoning_callback,
            }
        )
        if not self.responses:
            raise AssertionError("MockLLMClient has no more responses.")
        return self.responses.pop(0)


class MemoryHistoryStorage(HistoryStorage):
    """In-memory ``HistoryStorage``; records every saved snapshot in ``saves``."""

    def __init__(self, initial: HistorySnapshot | None = None):
        self.snapshot = initial if initial is not None else HistorySnapshot()
        self.saves: list[HistorySnapshot] = []

    async def load(self) -> HistorySnapshot:
        return self.snapshot

    async def save(self, snapshot: HistorySnapshot) -> None:
        self.snapshot = snapshot
        self.saves.append(snapshot)


def result_response(result: Any) -> LLMResponse:
    """A scripted "result" decision carrying ``result`` as the final answer.

    ``result`` may be anything the framework's JSON encoding handles —
    including dataclasses and Pydantic models, which serialize to the object
    shape the decision schema validates.
    """
    return LLMResponse(
        content=json.dumps(
            {"decision": "result", "result": result},
            default=pydantic_json_default,
        )
    )


def tool_calls_response(*calls: tuple[str, dict[str, Any]]) -> LLMResponse:
    """A scripted "tool_calls" decision from ``(tool_name, arguments)`` pairs."""
    return LLMResponse(
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
) -> AsyncIterator[Session]:
    """A ready-to-use sefia ``Session`` over an in-memory glyff backend.

    Pass ``backend`` (e.g. a shared ``MemoryBackend``) together with a stable
    ``session_id`` to simulate pause/resume across runs; by default each use
    gets a fresh backend, so nothing is replayed between tests. Extra keyword
    arguments go to ``Session`` (``tool_collector=``, ``profiles=``, ...).
    """
    async with glyff.Session(
        id=session_id,
        backend=backend if backend is not None else MemoryBackend(),
        serializer=PydanticSerializer(),
        hasher=PydanticArgsHasher(),
    ) as glyff_session:
        async with Session(
            llm_client=llm_client, glyff_session=glyff_session, **session_kwargs
        ) as session:
            yield session
