import asyncio
from typing import Any, cast

import pytest
from pytest_mock import MockerFixture

from sefia import ToolRegistry, current_tool_call_id
from sefia._tool_execution import call_tools
from sefia.event_system import EventPublisher
from sefia.exceptions import PauseException
from sefia.testing import make_tool_call_request


@pytest.fixture
def publisher(mocker: MockerFixture) -> EventPublisher:
    return cast(EventPublisher, mocker.AsyncMock(spec=EventPublisher))


async def test_concurrent_calls_overlap(publisher: EventPublisher) -> None:
    # The first tool blocks until the second one has run, which can only
    # complete if the batch is not serialized.
    peer_ran = asyncio.Event()

    async def wait_for_peer() -> str:
        await asyncio.wait_for(peer_ran.wait(), timeout=5)
        return "waited"

    async def release_peer() -> str:
        peer_ran.set()
        return "released"

    registry = ToolRegistry()
    registry.add(wait_for_peer, name="wait_for_peer", concurrent=True)
    registry.add(release_peer, name="release_peer", concurrent=True)

    results = await call_tools(
        [
            make_tool_call_request(id="1", name="wait_for_peer", arguments={}),
            make_tool_call_request(id="2", name="release_peer", arguments={}),
        ],
        registry,
        publisher,
    )

    # Results come back in request order even though the first call
    # finished last.
    assert [(r.tool_call_id, r.result) for r in results] == [
        ("1", "waited"),
        ("2", "released"),
    ]


async def test_unmarked_tools_stay_strictly_serial(publisher: EventPublisher) -> None:
    timeline: list[str] = []

    def make_tool(label: str):
        async def tool() -> str:
            timeline.append(f"{label}:start")
            await asyncio.sleep(0)
            timeline.append(f"{label}:end")
            return label

        return tool

    registry = ToolRegistry()
    registry.add(make_tool("a"), name="tool_a")
    registry.add(make_tool("b"), name="tool_b")

    await call_tools(
        [
            make_tool_call_request(id="1", name="tool_a", arguments={}),
            make_tool_call_request(id="2", name="tool_b", arguments={}),
        ],
        registry,
        publisher,
    )

    assert timeline == ["a:start", "a:end", "b:start", "b:end"]


async def test_serial_call_is_a_barrier_between_concurrent_calls(
    publisher: EventPublisher,
) -> None:
    timeline: list[str] = []

    def make_tool(label: str):
        async def tool() -> str:
            timeline.append(f"{label}:start")
            await asyncio.sleep(0)
            timeline.append(f"{label}:end")
            return label

        return tool

    registry = ToolRegistry()
    registry.add(make_tool("a"), name="conc_a", concurrent=True)
    registry.add(make_tool("s"), name="serial_s")
    registry.add(make_tool("b"), name="conc_b", concurrent=True)

    await call_tools(
        [
            make_tool_call_request(id="1", name="conc_a", arguments={}),
            make_tool_call_request(id="2", name="serial_s", arguments={}),
            make_tool_call_request(id="3", name="conc_b", arguments={}),
        ],
        registry,
        publisher,
    )

    assert timeline == [
        "a:start",
        "a:end",
        "s:start",
        "s:end",
        "b:start",
        "b:end",
    ]


async def test_pause_lets_concurrent_siblings_finish(
    publisher: EventPublisher,
) -> None:
    sibling_finished = False

    async def pausing() -> str:
        raise PauseException("needs input")

    async def sibling() -> str:
        nonlocal sibling_finished
        await asyncio.sleep(0)
        sibling_finished = True
        return "ok"

    registry = ToolRegistry()
    registry.add(pausing, name="pausing", concurrent=True)
    registry.add(sibling, name="sibling", concurrent=True)

    with pytest.raises(PauseException, match="needs input"):
        await call_tools(
            [
                make_tool_call_request(id="1", name="pausing", arguments={}),
                make_tool_call_request(id="2", name="sibling", arguments={}),
            ],
            registry,
            publisher,
        )

    assert sibling_finished


async def test_earliest_pause_in_request_order_wins(
    publisher: EventPublisher,
) -> None:
    # When several overlapped calls pause, the one earliest in request
    # order propagates, even if it was raised last in wall-clock time.
    second_paused = asyncio.Event()

    async def pause_late() -> str:
        await asyncio.wait_for(second_paused.wait(), timeout=5)
        raise PauseException("first in request order")

    async def pause_early() -> str:
        second_paused.set()
        raise PauseException("second in request order")

    registry = ToolRegistry()
    registry.add(pause_late, name="pause_late", concurrent=True)
    registry.add(pause_early, name="pause_early", concurrent=True)

    with pytest.raises(PauseException, match="first in request order"):
        await call_tools(
            [
                make_tool_call_request(id="1", name="pause_late", arguments={}),
                make_tool_call_request(id="2", name="pause_early", arguments={}),
            ],
            registry,
            publisher,
        )


async def test_tool_failure_in_concurrent_batch_stays_isolated(
    publisher: EventPublisher,
) -> None:
    # An ordinary failure is stringified into its own slot; siblings are
    # unaffected.
    async def boom() -> str:
        raise ValueError("kaboom")

    async def fine() -> str:
        return "ok"

    registry = ToolRegistry()
    registry.add(boom, name="boom", concurrent=True)
    registry.add(fine, name="fine", concurrent=True)

    results = await call_tools(
        [
            make_tool_call_request(id="1", name="boom", arguments={}),
            make_tool_call_request(id="2", name="fine", arguments={}),
        ],
        registry,
        publisher,
    )

    assert "Error executing tool 'boom'" in results[0].result
    assert results[1].result == "ok"


async def test_identical_concurrent_calls_run_serially(
    publisher: EventPublisher,
) -> None:
    # Identical calls (same tool and arguments) never overlap — glyff
    # sequences one content key by arrival order — while a call with
    # different arguments still does.
    active_same = 0
    max_active_same = 0
    saw_other_during_same = False
    active_other = 0

    async def fetch(key: str) -> str:
        nonlocal active_same, max_active_same, saw_other_during_same, active_other
        if key == "same":
            active_same += 1
            max_active_same = max(max_active_same, active_same)
        else:
            active_other += 1
        await asyncio.sleep(0.01)
        if key == "same" and active_other:
            saw_other_during_same = True
        if key == "same":
            active_same -= 1
        else:
            active_other -= 1
        return key

    registry = ToolRegistry()
    registry.add(fetch, name="fetch", concurrent=True)

    await call_tools(
        [
            make_tool_call_request(id="1", name="fetch", arguments={"key": "same"}),
            make_tool_call_request(id="2", name="fetch", arguments={"key": "same"}),
            make_tool_call_request(id="3", name="fetch", arguments={"key": "other"}),
        ],
        registry,
        publisher,
    )

    assert max_active_same == 1
    assert saw_other_during_same


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


async def test_call_id_is_inherited_by_a_task_spawned_during_the_call(
    publisher: EventPublisher,
) -> None:
    # A spawned task copies the context, so it reads the id past the handler's
    # return — the inheritance the accessor's contract documents.
    seen: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def background() -> None:
        await asyncio.sleep(0)
        seen.set_result(current_tool_call_id())

    def spawns_background() -> str:
        asyncio.ensure_future(background())
        return "ok"

    registry = ToolRegistry()
    registry.add(spawns_background, name="spawns_background")

    await call_tools(
        [make_tool_call_request(id="bg-call", name="spawns_background", arguments={})],
        registry,
        publisher,
    )

    assert await asyncio.wait_for(seen, timeout=1) == "bg-call"
