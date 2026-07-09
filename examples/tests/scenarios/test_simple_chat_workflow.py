from importlib import import_module
from unittest.mock import AsyncMock

import pytest

from sefios.cli import SefiaCLI, SefiaCLISession

main = import_module("examples.00_simple_chat.main")


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    """Point the simple chat example at a throwaway session directory."""
    cli = SefiaCLI(session_dir=tmp_path / "sessions", model="gpt-4o-mini", stream=False)
    monkeypatch.setattr(main, "sefia_cli", cli)
    monkeypatch.setattr(main, "agent", main.ChatAgent(cli.input_tool))
    return main


class TestSimpleChatWorkflow:
    async def test_accepts_input_and_runs_chat_agent(self, workflow, monkeypatch):
        accepted_inputs = []
        original_accept_input = SefiaCLISession.accept_input

        async def accept_input_spy(self, input_value, *, reply_to=None):
            accepted_inputs.append((input_value, reply_to))
            await original_accept_input(self, input_value, reply_to=reply_to)

        chat = AsyncMock()
        monkeypatch.setattr(SefiaCLISession, "accept_input", accept_input_spy)
        monkeypatch.setattr(workflow.agent, "chat", chat)

        await workflow.chat.__wrapped__(
            message=["hello", "there"],
            model="gpt-4o-mini",
        )

        assert accepted_inputs == [(["hello", "there"], None)]
        chat.assert_awaited_once()
        assert workflow.sefia_cli.get_active_session() is not None

    async def test_reuses_active_session_for_next_message(self, workflow, monkeypatch):
        chat = AsyncMock()
        monkeypatch.setattr(workflow.agent, "chat", chat)

        await workflow.chat.__wrapped__(
            message=["first"],
            model="gpt-4o-mini",
        )
        first_session = workflow.sefia_cli.get_active_session()

        await workflow.chat.__wrapped__(
            message=["second"],
            model="gpt-4o-mini",
        )

        assert workflow.sefia_cli.get_active_session() == first_session
        assert chat.await_count == 2
