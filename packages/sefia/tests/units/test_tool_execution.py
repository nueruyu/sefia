from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture
from sefia import ToolRegistry, current_tool_call_id
from sefia._tool_execution import call_tools
from sefia.event_system import EventPublisher
from sefia.events import AfterToolCall, BeforeToolCall, ToolExecutionFailed
from sefia.inference import ToolCallResult
from sefia.testing import make_tool_call_request


@pytest.fixture
def publisher(mocker: MockerFixture) -> EventPublisher:
    return cast(EventPublisher, mocker.AsyncMock(spec=EventPublisher))


async def test_handler_reads_its_own_call_id(publisher: EventPublisher) -> None:
    # Each handler sees the id of the call it is serving, for a signature-based
    # tool and a JSON-schema tool alike, and distinct calls see distinct ids.
    seen: dict[str, str] = {}

    def note_signature(label: str) -> str:
        seen[label] = current_tool_call_id()
        return "ok"

    def note_json(**arguments: Any) -> str:
        seen[arguments["label"]] = current_tool_call_id()
        return "ok"

    registry = ToolRegistry()
    registry.add(note_signature, name="note_signature")
    registry.add_json_tool(
        note_json,
        name="note_json",
        description="",
        parameters={"type": "object"},
    )

    await call_tools(
        [
            make_tool_call_request(
                id="sig-call", name="note_signature", arguments={"label": "a"}
            ),
            make_tool_call_request(
                id="json-call", name="note_json", arguments={"label": "b"}
            ),
        ],
        registry,
        publisher,
    )

    assert seen == {"a": "sig-call", "b": "json-call"}


async def test_call_id_is_unbound_in_the_caller_after_the_call_returns(
    publisher: EventPublisher,
) -> None:
    registry = ToolRegistry()
    registry.add(lambda: "ok", name="noop")

    await call_tools(
        [make_tool_call_request(id="1", name="noop", arguments={})],
        registry,
        publisher,
    )

    with pytest.raises(RuntimeError):
        current_tool_call_id()


@pytest.mark.parametrize("exists", [False, True])
async def test_tool_failure_becomes_result_and_failure_event(exists: bool) -> None:
    publisher = AsyncMock(spec=EventPublisher)
    registry = ToolRegistry()
    error = ValueError("kaboom")
    if exists:
        registry.add(AsyncMock(side_effect=error), name="boom")
    call = make_tool_call_request(id="1", name="boom", arguments={})

    results = await call_tools([call], registry, publisher)

    expected = (
        "Error executing tool 'boom': ValueError(kaboom)"
        if exists
        else "Error: Tool 'boom' not found."
    )
    assert results == [ToolCallResult(tool_call_id="1", result=expected)]
    before, failed = [c.args[0] for c in publisher.publish.await_args_list]
    assert before == BeforeToolCall(tool_call=call)
    assert isinstance(failed, ToolExecutionFailed)
    assert failed.tool_call is call
    if exists:
        assert failed.error is error
    else:
        assert isinstance(failed.error, RuntimeError)
    with pytest.raises(RuntimeError):
        current_tool_call_id()


async def test_successful_tool_publishes_result() -> None:
    registry = ToolRegistry()
    registry.add(lambda: "ok", name="tool")
    publisher = AsyncMock(spec=EventPublisher)
    call = make_tool_call_request(name="tool")

    assert await call_tools([call], registry, publisher) == [
        ToolCallResult(tool_call_id=call.id, result="ok")
    ]
    assert [c.args[0] for c in publisher.publish.await_args_list] == [
        BeforeToolCall(tool_call=call),
        AfterToolCall(tool_call=call, result="ok"),
    ]
