import asyncio
from pathlib import Path

import pytest
from sefia.testing import MockLLMClient
from sefios import FilePersistence, SQLitePersistence, domain, require_input
from sefios.exceptions import (
    InteractionConflictError,
    InteractionRequired,
    UnknownInteractionError,
)
from sefios.fastapi import SefiaHTTP

engrave = domain(__name__).engrave


@engrave
async def two_questions() -> tuple[str, str]:
    return await require_input("Continue?"), await require_input("Continue?")


@pytest.mark.parametrize("backend", ["sqlite", "file"])
async def test_repeated_input_survives_restart(tmp_path: Path, backend: str) -> None:
    def app() -> SefiaHTTP:
        persistence = (
            SQLitePersistence(tmp_path / "sessions.db")
            if backend == "sqlite"
            else FilePersistence(tmp_path)
        )
        return SefiaHTTP(llm_client=MockLLMClient([]), persistence=persistence)

    first = app()
    sid = first.create_session()
    with pytest.raises(InteractionRequired) as pause:
        async with first.session(session_id=sid):
            await two_questions()
    first_id = pause.value.interaction_id
    assert first_id is not None

    second = app()
    with pytest.raises(InteractionRequired) as repeated:
        async with second.session(session_id=sid):
            await two_questions()
    assert repeated.value.interaction_id == first_id

    with pytest.raises(InteractionRequired) as next_pause:
        async with second.session(session_id=sid) as session:
            await session.resolve_interaction(first_id, "first")
            await two_questions()
    second_id = next_pause.value.interaction_id
    assert second_id is not None and second_id != first_id

    third = app()
    async with third.session(session_id=sid) as session:
        await session.resolve_interaction(second_id, "second")
        assert await two_questions() == ("first", "second")
    async with app().session(session_id=sid):
        assert await two_questions() == ("first", "second")


async def test_unknown_future_interaction_is_rejected() -> None:
    http = SefiaHTTP(llm_client=MockLLMClient([]))
    async with http.session(session_id=http.create_session()) as session:
        with pytest.raises(UnknownInteractionError):
            await session.resolve_interaction("future", "yes")
        assert await session.pending_interactions() == []


async def test_parallel_pending_routes_each_reply(tmp_path: Path) -> None:
    http = SefiaHTTP(
        llm_client=MockLLMClient([]),
        persistence=SQLitePersistence(tmp_path / "sessions.db"),
    )
    sid = http.create_session()
    async with http.session(session_id=sid):
        pauses = await asyncio.gather(
            require_input("A?"), require_input("B?"), return_exceptions=True
        )
    assert all(isinstance(p, InteractionRequired) for p in pauses)
    ids = [p.interaction_id for p in pauses if isinstance(p, InteractionRequired)]
    assert len(set(ids)) == 2
    async with http.session(session_id=sid) as session:
        await session.resolve_interaction(ids[1], "b")
        await session.resolve_interaction(ids[0], "a")
        assert await asyncio.gather(require_input("A?"), require_input("B?")) == [
            "a",
            "b",
        ]
        with pytest.raises(InteractionConflictError):
            await session.resolve_interaction(ids[0], "duplicate")


async def test_tool_and_application_requests_share_reply_routing() -> None:
    from sefia import ToolRegistry
    from sefia._tool_execution import call_tools
    from sefia.event_system import EventPublisher
    from sefia.testing import make_tool_call_request

    http = SefiaHTTP(llm_client=MockLLMClient([]))
    sid = http.create_session()
    registry = ToolRegistry()
    registry.add(http.input_tool.get_input, name="input")

    async def tool_input() -> str:
        results = await call_tools(
            [
                make_tool_call_request(
                    id="tool-call", name="input", arguments={"prompt": "Tool?"}
                )
            ],
            registry,
            EventPublisher([]),
        )
        return str(results[0].result)

    async with http.session(session_id=sid):
        with pytest.raises(InteractionRequired) as app_pause:
            await require_input("Application?")
        with pytest.raises(InteractionRequired) as tool_pause:
            await tool_input()
    assert tool_pause.value.interaction_id == "tool-call"
    assert app_pause.value.interaction_id != tool_pause.value.interaction_id
    async with http.session(session_id=sid) as session:
        await session.resolve_interaction(
            tool_pause.value.interaction_id, "tool answer"
        )
        await session.resolve_interaction(app_pause.value.interaction_id, "app answer")
        assert await require_input("Application?") == "app answer"
        assert await tool_input() == "tool answer"


async def test_cli_uses_same_application_input_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typer
    from sefia_litellm import LiteLLMClient
    from sefios.cli import SefiaCLI

    client = MockLLMClient([])
    monkeypatch.setattr(LiteLLMClient, "complete", client.complete)
    cli = SefiaCLI(model="unused", reporter=None)
    sid = cli.create_session()
    with pytest.raises(typer.Exit) as exit_info:
        async with cli.session(session_id=sid):
            await require_input("Continue?")
    assert exit_info.value.exit_code == 0
    async with cli.session(session_id=sid) as session:
        pending = await session.pending_interactions()
        assert len(pending) == 1
        await session.resolve_interaction(pending[0].interaction_id, "yes")
        assert await require_input("Continue?") == "yes"
    assert client.requests == []
