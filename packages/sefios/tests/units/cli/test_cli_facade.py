import pytest
from sefia_typer.exceptions import UnknownSessionError as CLIUnknownSessionError
from sefios import MemoryPersistence
from sefios.cli import SefiaCLI, SefiaCLISession
from sefios.interactions import InteractionChannel, InteractionResult
from sefios.storage import MemorySessionStorage
from sefios.tools import Input, Output


async def test_cli_generic_resolution(
    memory_session_storage: MemorySessionStorage,
) -> None:
    channel = InteractionChannel(memory_session_storage)
    session = SefiaCLISession(channel=channel)
    await channel.request("a", {"type": "input", "prompt": "q?"})
    assert len(await session.pending_interactions()) == 1
    await session.resolve_interaction("a", "answer")
    assert await session.pending_interactions() == []
    assert await channel.request(
        "a", {"type": "input", "prompt": "q?"}
    ) == InteractionResult("answer")


class TestSefiaCLISessionManagement:
    @pytest.fixture
    def cli(self) -> SefiaCLI:
        return SefiaCLI(model="gpt-4o")

    def test_input_tool_is_exposed(self, cli: SefiaCLI):
        assert isinstance(cli.input_tool, Input)

    def test_output_tool_is_exposed(self, cli: SefiaCLI):
        assert isinstance(cli.output_tool, Output)

    def test_create_session_becomes_active(self, cli: SefiaCLI):
        session_id = cli.create_session()

        assert cli.get_active_session() == session_id

    def test_switch_session(self, cli: SefiaCLI):
        first = cli.create_session()
        second = cli.create_session()
        assert cli.get_active_session() == second

        switched = cli.switch_session(first)

        assert switched == first
        assert cli.get_active_session() == first

    def test_switch_to_unknown_session_raises_cli_error(self, cli: SefiaCLI):
        # The facade translates the sefios-internal exception into the
        # sefia_typer one that applications catch.
        with pytest.raises(CLIUnknownSessionError) as exc_info:
            cli.switch_session("ghost")

        assert exc_info.value.session_id == "ghost"

    def test_no_active_session_initially(self, cli: SefiaCLI):
        assert cli.get_active_session() is None

    def test_default_active_selection_is_process_local(self) -> None:
        first = SefiaCLI(model="gpt-4o")
        session_id = first.create_session()

        second = SefiaCLI(model="gpt-4o")

        assert first.get_active_session() == session_id
        assert second.get_active_session() is None

    def test_explicit_memory_persistence_keeps_active_selection_in_memory(
        self,
    ) -> None:
        first = SefiaCLI(model="gpt-4o", persistence=MemoryPersistence())
        session_id = first.create_session()

        second = SefiaCLI(model="gpt-4o", persistence=MemoryPersistence())

        assert first.get_active_session() == session_id
        assert second.get_active_session() is None

    def test_registry_is_shared_but_active_selection_is_local(self) -> None:
        persistence = MemoryPersistence()
        first = SefiaCLI(model="gpt-4o", persistence=persistence)
        second = SefiaCLI(model="gpt-4o", persistence=persistence)

        session_id = first.create_session()

        assert second.get_active_session() is None
        assert second.switch_session(session_id) == session_id
