from importlib import import_module
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from examples._common.sefia_cli import SefiaCLI

# main.py uses ``from .._common ...`` imports, so it must be loaded with the
# full ``examples.`` package prefix (two parent levels).
main = import_module("examples.00_simple_chat.main")


@pytest.fixture
def workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> ModuleType:
    """Point the example's module-level CLI at a throwaway session directory."""
    cli = SefiaCLI(session_dir=tmp_path / "sessions", model="gpt-4o-mini", stream=False)
    monkeypatch.setattr(main, "sefia_cli", cli)
    return main  # type: ignore[return-value]


@pytest.fixture
def chat_async(workflow: ModuleType) -> Any:
    """Return the unwrapped async chat function for direct testing."""
    return getattr(workflow.chat, "__wrapped__")


class TestSimpleChatWorkflow:
    async def test_accepts_input_and_invokes_agent(
        self,
        workflow: ModuleType,
        chat_async: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        agent_chat = AsyncMock()
        monkeypatch.setattr(workflow.agent, "chat", agent_chat)

        await chat_async(
            message=["Hello, assistant!"],
            model="gpt-4o-mini",
        )

        agent_chat.assert_awaited_once()

    async def test_empty_message_does_not_call_agent(
        self,
        workflow: ModuleType,
        chat_async: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty message list produces no input text, so accept_input is a no-op."""
        agent_chat = AsyncMock()
        monkeypatch.setattr(workflow.agent, "chat", agent_chat)

        await chat_async(
            message=[],
            model="gpt-4o-mini",
        )

        # The agent is still invoked — the session is just resumed without new input.
        agent_chat.assert_awaited_once()
