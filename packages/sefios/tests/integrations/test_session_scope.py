from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import glyff
import pytest
import sefia
from sefia import (
    DecisionContext,
    DecisionMiddleware,
    JsonSchemaToolEntry,
    MiddlewareSet,
    Policy,
)
from sefia.exceptions import InferenceError
from sefia.inference import FunctionInfo, ResultDecision, StepDecision
from sefia.llm import LLMCompletion, Message, MessageComposer, MessageLayout
from sefia.pydantic import (
    PydanticJsonMaterializer,
    PydanticResultFormatFactory,
)
from sefia.testing import MockLLMClient, result_completion, tool_calls_completion
from sefia.tool_collectors import StaticToolCollector
from sefios import (
    MemoryPersistence,
    MemorySessionStorage,
    SessionScope,
    SQLitePersistence,
    SQLiteSessionStorage,
    domain,
    get_session_storage,
)
from sefios.middleware import Retrier
from typing_extensions import final, override

infer = domain("packages.sefios.tests.integrations.test_session_scope").infer


class _Prefix(MessageComposer):
    def __init__(self, text: str) -> None:
        self.text = text

    @override
    async def compose(
        self, function: FunctionInfo, layout: MessageLayout
    ) -> MessageLayout:
        return MessageLayout(
            before=(Message(role="developer", content=self.text), *layout.before),
            arguments=layout.arguments,
            after=layout.after,
        )


class _Probe:
    """Receiver used to exercise the configured tool collector."""

    @infer
    async def answer(self) -> str:
        """Answer using the available tools."""
        ...


def _static_collector(name: str, calls: list[str]) -> StaticToolCollector:
    async def handler() -> str:
        calls.append(name)
        return f"{name}-result"

    return StaticToolCollector(
        [
            JsonSchemaToolEntry(
                handler,
                name=name,
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                description=f"the {name} tool",
            )
        ]
    )


async def test_tool_collector_default_is_used(
    make_mock_llm: Callable[[list[LLMCompletion]], MockLLMClient],
) -> None:
    calls: list[str] = []
    llm = make_mock_llm(
        [tool_calls_completion(("init_tool", {})), result_completion("ok")]
    )
    scope = SessionScope(
        llm_client=llm,
        tool_collector=_static_collector("init_tool", calls),
        persistence=MemoryPersistence(),
    )

    async with scope.session(session_id="s"):
        assert await _Probe().answer() == "ok"

    assert calls == ["init_tool"]


async def test_session_tool_collector_overrides_init_default(
    make_mock_llm: Callable[[list[LLMCompletion]], MockLLMClient],
) -> None:
    calls: list[str] = []
    llm = make_mock_llm(
        [tool_calls_completion(("call_tool", {})), result_completion("ok")]
    )
    scope = SessionScope(
        llm_client=llm,
        tool_collector=_static_collector("init_tool", calls),
        persistence=MemoryPersistence(),
    )

    async with scope.session(
        session_id="s", tool_collector=_static_collector("call_tool", calls)
    ):
        assert await _Probe().answer() == "ok"

    assert calls == ["call_tool"]


async def test_session_message_composers_override_scope_default() -> None:
    client = MockLLMClient([result_completion("first"), result_completion("second")])
    scope = SessionScope(llm_client=client, message_composers=(_Prefix("scope"),))

    async with scope.session(session_id="composer-default"):
        assert await _Probe().answer() == "first"
    async with scope.session(
        session_id="composer-override", message_composers=(_Prefix("session"),)
    ):
        assert await _Probe().answer() == "second"

    assert [request["messages"][0]["content"] for request in client.requests] == [
        "scope",
        "session",
    ]


async def test_strategy_capabilities_inherit_and_override_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope_result_factory = PydanticResultFormatFactory()
    scope_converter = PydanticJsonMaterializer()
    session_result_factory = PydanticResultFormatFactory()
    session_converter = PydanticJsonMaterializer()
    captured: list[dict[str, Any]] = []

    class _RecordingSession:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

        async def __aenter__(self) -> "_RecordingSession":
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: object | None,
        ) -> None:
            pass

    monkeypatch.setattr(sefia, "Session", _RecordingSession)
    scope = SessionScope(
        llm_client=MockLLMClient([]),
        result_format_factory=scope_result_factory,
        json_materializer=scope_converter,
    )

    async with scope.session(session_id="scope-capabilities"):
        pass
    async with scope.session(
        session_id="result-factory-override",
        result_format_factory=session_result_factory,
    ):
        pass
    async with scope.session(
        session_id="converter-override",
        json_materializer=session_converter,
    ):
        pass

    assert captured[0]["result_format_factory"] is scope_result_factory
    assert captured[0]["json_materializer"] is scope_converter
    assert captured[1]["result_format_factory"] is session_result_factory
    assert captured[1]["json_materializer"] is scope_converter
    assert captured[2]["result_format_factory"] is scope_result_factory
    assert captured[2]["json_materializer"] is session_converter


async def test_memory_persistence_is_default(
    make_mock_llm: Callable[[list[LLMCompletion]], MockLLMClient],
) -> None:
    scope = SessionScope(llm_client=make_mock_llm([]))

    async with scope.session(session_id="custom-store"):
        assert isinstance(get_session_storage(), MemorySessionStorage)


async def test_sqlite_persistence_can_be_selected(
    tmp_path: Path,
    make_mock_llm: Callable[[list[LLMCompletion]], MockLLMClient],
) -> None:
    database = tmp_path / "sessions.sqlite3"
    scope = SessionScope(
        llm_client=make_mock_llm([]),
        persistence=SQLitePersistence(database),
    )

    async with scope.session(session_id="durable"):
        assert isinstance(get_session_storage(), SQLiteSessionStorage)

    assert database.is_file()


async def test_retrier_regenerates_final_decision_after_committed_tools() -> None:
    persistence = MemoryPersistence()
    backend = persistence.create_execution_backend()
    session_id = "decision-retry"
    entries: list[tuple[int, glyff.ExecutionId]] = []
    tool_calls: list[str] = []

    @final
    class RejectFirstResult(DecisionMiddleware):
        @override
        async def wrap(
            self, ctx: DecisionContext, nxt: Callable[[], Awaitable[StepDecision]]
        ) -> StepDecision:
            execution_id = glyff.get_context().current_execution_id
            assert execution_id is not None
            execution = await backend.repository.get(
                glyff.SessionId(session_id), execution_id
            )
            assert execution is not None
            assert execution.status is glyff.ExecutionStatus.STARTED
            assert execution.result is None
            entries.append((ctx.step, execution_id))
            decision = await nxt()
            if decision == ResultDecision("rejected"):
                raise InferenceError("retry this decision")
            return decision

    client = MockLLMClient(
        [
            tool_calls_completion(("lookup", {})),
            result_completion("rejected"),
            result_completion("accepted"),
        ]
    )
    scope = SessionScope(
        llm_client=client,
        persistence=persistence,
        tool_collector=_static_collector("lookup", tool_calls),
        policies=[
            Policy(
                middleware=lambda: MiddlewareSet(
                    inference=(Retrier(max_retries=1),), decision=(RejectFirstResult(),)
                )
            )
        ],
    )
    async with scope.session(session_id=session_id):
        assert await _Probe().answer() == "accepted"

    assert [step for step, _ in entries] == [0, 1, 1]
    rejected_id, accepted_id = entries[1][1], entries[2][1]
    assert rejected_id.parent_id == accepted_id.parent_id
    assert rejected_id.name == accepted_id.name
    assert rejected_id.arguments_digest == accepted_id.arguments_digest
    # An in-process retry advances Glyff's sequence within the same parent call.
    assert accepted_id.sequence == rejected_id.sequence + 1
    assert tool_calls == ["lookup"]
    assert len(client.requests) == 3
    assert client.requests[-1]["messages"] == client.requests[-2]["messages"]
    assert "lookup-result" in str(client.requests[-1]["messages"])
    executions = [
        execution
        async for execution in backend.repository.executions(
            glyff.SessionId(session_id)
        )
    ]
    assert len([e for e in executions if e.id.name.value == "inference.step"]) == 3
    rejected = next(e for e in executions if e.id == rejected_id)
    assert rejected.status is glyff.ExecutionStatus.STARTED
    assert rejected.result is None
    assert all(
        e.status is glyff.ExecutionStatus.COMPLETED and e.result is not None
        for e in executions
        if e.id != rejected_id
    )

    async with scope.session(session_id=session_id):
        assert await _Probe().answer() == "accepted"
    assert len(entries) == 3
    assert len(client.requests) == 3
    assert tool_calls == ["lookup"]
