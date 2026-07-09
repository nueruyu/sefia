import pytest
from glyff_pydantic import PydanticSerializer
from sefia_typer import InputReceiver, InputStore
from sefia_typer import UnknownSessionError as CLIUnknownSessionError
from sefios.cli import CostReportingCLIReporter, SefiaCLI, SefiaCLISession
from sefios.cli._app import _USE_DEFAULT_REPORTER
from sefios.storage import MemorySessionStorage
from sefios.tools import InputTool


@pytest.fixture
def session_storage() -> MemorySessionStorage:
    return MemorySessionStorage(serializer=PydanticSerializer())


class TestSefiaCLISession:
    @pytest.fixture
    def store(self) -> InputStore:
        return InputStore()

    @pytest.fixture
    def session(self, store, session_storage):
        with store.use_store(session_storage):
            receiver = InputReceiver(store)
            yield SefiaCLISession(input_receiver=receiver)

    async def test_none_input_is_ignored(self, session, store):
        await session.accept_input(None)

        assert await store.pop_queued_input() is None

    async def test_string_input_is_stored(self, session, store):
        await session.accept_input("hello")

        assert await store.pop_queued_input() == "hello"

    async def test_list_input_is_joined_and_stored(self, session, store):
        await session.accept_input(["hello", "world"])

        assert await store.pop_queued_input() == "hello world"

    async def test_reply_to_answers_pending_request(self, session, store):
        await store.save_pending_requests({"a": {"id": "a", "prompt": "q?"}})

        await session.accept_input("answer", reply_to="a")

        assert await store.get_input("a") == "answer"


class TestSefiaCLISessionManagement:
    @pytest.fixture
    def cli(self, tmp_path) -> SefiaCLI:
        return SefiaCLI(session_dir=tmp_path / "sessions", model="gpt-4o")

    def test_input_tool_is_exposed(self, cli: SefiaCLI):
        assert isinstance(cli.input_tool, InputTool)

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
        # sefia_typer one that `add_session_commands` handles.
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
