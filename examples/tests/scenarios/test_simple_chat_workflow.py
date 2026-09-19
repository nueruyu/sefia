from importlib import import_module
from types import ModuleType
from unittest.mock import AsyncMock

import pytest
from sefios.cli import SefiaCLI

main = import_module("examples.00_simple_chat.main")


@pytest.fixture
def workflow(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    cli = SefiaCLI(model="gpt-4o-mini", stream=False)
    monkeypatch.setattr(main, "sefia_cli", cli)
    monkeypatch.setattr(main, "agent", main.ChatAgent(cli.input_tool, cli.output_tool))
    return main


class TestSimpleChatWorkflow:
    async def test_starts_without_queued_input(
        self, workflow: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat = AsyncMock()
        monkeypatch.setattr(workflow.agent, "chat", chat)
        await workflow.chat.__wrapped__(
            message=None, interaction_id=None, model="gpt-4o-mini"
        )
        chat.assert_awaited_once()
        assert workflow.sefia_cli.get_active_session() is not None

    async def test_rejects_message_without_interaction_id(
        self, workflow: ModuleType
    ) -> None:
        import typer

        with pytest.raises(typer.BadParameter):
            await workflow.chat.__wrapped__(
                message=["hello"], interaction_id=None, model="gpt-4o-mini"
            )
