import pytest
from glyff_pydantic import PydanticSerializer
from sefia_typer import InputChannel
from sefia_typer import UnknownSessionError as CLIUnknownSessionError
from sefios.cli import CostReportingCLIReporter, SefiaCLI, SefiaCLISession
from sefios.cli._app import _USE_DEFAULT_REPORTER
from sefios.storage import MemorySessionStorage
from sefios.tools import Input, Output


@pytest.fixture
def session_storage() -> MemorySessionStorage:
    return MemorySessionStorage(serializer=PydanticSerializer())


class TestSefiaCLISession:
    @pytest.fixture
    def channel(self) -> InputChannel:
        return InputChannel()

    @pytest.fixture
    def session(self, channel, session_storage):
        # A bound SessionStorage satisfies the adapter's KeyValueStore protocol.
        with channel.use_store(session_storage):
            yield SefiaCLISession(channel=channel)

    async def test_none_input_is_ignored(self, session, channel):
        await session.accept_input(None)

        assert await channel.provide_input("any") is None

    async def test_string_input_is_stored(self, session, channel):
        await session.accept_input("hello")

        assert await channel.provide_input("any") == "hello"

    async def test_list_input_is_joined_and_stored(self, session, channel):
        await session.accept_input(["hello", "world"])

        assert await channel.provide_input("any") == "hello world"

    async def test_reply_to_resolves_pending_request(self, session, channel):
        await channel.record_request("a", "q?")

        await session.accept_input("answer", reply_to="a")

        assert await channel.provide_input("a") == "answer"


class TestSefiaCLISessionManagement:
    @pytest.fixture
    def cli(self, tmp_path) -> SefiaCLI:
        return SefiaCLI(session_dir=tmp_path / "sessions", model="gpt-4o")

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


class TestReporterResolution:
    def test_default_sentinel_resolves_to_cost_reporting_reporter(self, tmp_path):
        cli = SefiaCLI(
            session_dir=tmp_path / "sessions",
            model="gpt-4o",
            reporter=_USE_DEFAULT_REPORTER,
        )

        assert isinstance(cli._reporter, CostReportingCLIReporter)

    def test_explicit_none_disables_reporting(self, tmp_path):
        cli = SefiaCLI(
            session_dir=tmp_path / "sessions",
            model="gpt-4o",
            reporter=None,
        )

        assert cli._reporter is None
