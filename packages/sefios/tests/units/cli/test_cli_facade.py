from collections.abc import Iterator

import pytest
from sefia_typer.exceptions import UnknownSessionError as CLIUnknownSessionError
from sefios import MemoryPersistence
from sefios._input_channel import InputChannel
from sefios.cli import SefiaCLI, SefiaCLISession
from sefios.storage import MemorySessionStorage
from sefios.tools import Input, Output


class TestSefiaCLISession:
    @pytest.fixture
    def channel(self) -> InputChannel:
        return InputChannel()

    @pytest.fixture
    def session(
        self,
        channel: InputChannel,
        memory_session_storage: MemorySessionStorage,
    ) -> Iterator[SefiaCLISession]:
        # A bound SessionStorage satisfies the input channel's store protocol.
        with channel.use_store(memory_session_storage):
            yield SefiaCLISession(channel=channel)

    async def test_reply_to_resolves_pending_request(
        self, session: SefiaCLISession, channel: InputChannel
    ) -> None:
        await channel.record_request("a", "q?")

        await session.accept_input("answer", reply_to="a")

        assert await channel.provide_input("a") == "answer"


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
