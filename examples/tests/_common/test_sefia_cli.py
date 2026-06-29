import pytest

from examples._common.human_input import (
    CLIHumanInputReceiver,
    HumanInputSessionStore,
)
from examples._common.sefia_cli import (
    DefaultCLIReporter,
    SefiaCLI,
    SefiaCLISession,
    _to_input_text,
    _USE_DEFAULT_REPORTER,
)
from examples._common.session import ResolvedSession
from sefia.exceptions import InvalidInferenceResponseError
from sefios.tools import HumanInputRequest, HumanInputTool


class TestToInputText:
    def test_plain_string_is_stripped(self):
        assert _to_input_text("  hello  ") == "hello"

    def test_list_is_joined_with_spaces(self):
        assert _to_input_text(["hello", "world"]) == "hello world"

    def test_list_result_is_stripped(self):
        assert _to_input_text(["  hello  "]) == "hello"

    def test_empty_list_is_empty_string(self):
        assert _to_input_text([]) == ""


class TestSefiaCLISession:
    @pytest.fixture
    def store(self) -> HumanInputSessionStore:
        return HumanInputSessionStore()

    @pytest.fixture
    def session(self, store, session_store):
        with store.use_session_store(session_store):
            receiver = CLIHumanInputReceiver(store)
            yield SefiaCLISession(human_input=receiver)

    async def test_none_input_is_ignored(self, session, store):
        await session.accept_input(None)

        assert await store.pop_queued_input() is None

    async def test_blank_input_is_ignored(self, session, store):
        await session.accept_input("   ")

        assert await store.pop_queued_input() is None

    async def test_string_input_is_stored(self, session, store):
        await session.accept_input("hello")

        assert await store.pop_queued_input() == "hello"

    async def test_list_input_is_joined_and_stored(self, session, store):
        await session.accept_input(["hello", "world"])

        assert await store.pop_queued_input() == "hello world"

    async def test_reply_to_answers_pending_request(self, session, store):
        await store.save_pending_requests({"a": {"id": "a", "question": "q?"}})

        await session.accept_input("answer", reply_to="a")

        assert await store.get_answer("a") == "answer"


class TestSefiaCLISessionManagement:
    @pytest.fixture
    def cli(self, tmp_path) -> SefiaCLI:
        return SefiaCLI(session_dir=tmp_path / "sessions", model="gpt-4o")

    def test_human_input_tool_is_exposed(self, cli: SefiaCLI):
        assert isinstance(cli.human_input_tool, HumanInputTool)

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

    def test_no_active_session_initially(self, cli: SefiaCLI):
        assert cli.get_active_session() is None


class TestReporterResolution:
    def test_default_sentinel_resolves_to_default_reporter(self, tmp_path):
        cli = SefiaCLI(
            session_dir=tmp_path / "sessions",
            model="gpt-4o",
            reporter=_USE_DEFAULT_REPORTER,
        )

        assert isinstance(cli._reporter, DefaultCLIReporter)

    def test_explicit_none_disables_reporting(self, tmp_path):
        cli = SefiaCLI(
            session_dir=tmp_path / "sessions",
            model="gpt-4o",
            reporter=None,
        )

        assert cli._reporter is None


class TestDefaultCLIReporter:
    def test_created_session_is_announced(self, capsys):
        reporter = DefaultCLIReporter()

        reporter.on_session_resolved(
            ResolvedSession(session_id="abc", is_new=True, source="created")
        )

        assert "abc" in capsys.readouterr().out

    def test_active_session_is_announced(self, capsys):
        reporter = DefaultCLIReporter()

        reporter.on_session_resolved(
            ResolvedSession(session_id="abc", is_new=False, source="active")
        )

        assert "abc" in capsys.readouterr().out

    def test_explicit_session_is_quiet(self, capsys):
        reporter = DefaultCLIReporter()

        reporter.on_session_resolved(
            ResolvedSession(session_id="abc", is_new=False, source="explicit")
        )

        assert capsys.readouterr().out == ""

    def test_human_input_request_includes_marker(self, capsys):
        reporter = DefaultCLIReporter()

        reporter.on_human_input_request(
            HumanInputRequest(interaction_id="xyz", question="What topic?")
        )

        output = capsys.readouterr().out
        assert "USER_INPUT_REQUIRED:xyz" in output
        assert "What topic?" in output

    def test_human_input_question_delta_is_printed_without_newline(self, capsys):
        reporter = DefaultCLIReporter()

        reporter.on_human_input_question_delta("What ")
        reporter.on_human_input_question_delta("topic?")

        assert capsys.readouterr().out == "What topic?"

    async def test_inference_error_is_reported_as_error(self, capsys, mocker):
        reporter = DefaultCLIReporter()
        mocker.patch.object(reporter, "_echo_total_cost")

        await reporter.on_inference_error(
            InvalidInferenceResponseError("bad model response")
        )

        output = capsys.readouterr().out
        assert "INFERENCE ERROR" in output
        assert "bad model response" in output
        assert "WAITING FOR INPUT" not in output
