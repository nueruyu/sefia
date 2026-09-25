import pytest
from sefia.exceptions import InferenceError
from sefia_typer import (
    CLIReporter,
    InteractionRequest as CLIInteractionRequest,
    OutputMessage as CLIOutputMessage,
    ResolvedSession as CLIResolvedSession,
)
from sefia_typer.exceptions import UnknownSessionError as CLIUnknownSessionError
from sefios import MemoryPersistence
from sefios.cli import SefiaCLI, SefiaCLISession
from sefios.interactions import InteractionChannel, InteractionResult
from sefios.storage import MemorySessionStorage
from sefios.tools import Input, Output


class _RecordingCLIReporter(CLIReporter):
    def __init__(self) -> None:
        self.resolved_sessions: list[tuple[str, str]] = []

    def on_session_resolved(self, session: CLIResolvedSession) -> None:
        self.resolved_sessions.append((session.session_id, session.source))

    def on_interaction_request(self, request: CLIInteractionRequest) -> None:
        pass

    def on_input_prompt_delta(self, interaction_id: str, text: str) -> None:
        pass

    def on_output(self, message: CLIOutputMessage) -> None:
        pass

    def on_output_message_delta(self, interaction_id: str, text: str) -> None:
        pass

    def on_interrupted(self, session: CLIResolvedSession) -> None:
        pass

    def on_inference_error(self, error: InferenceError) -> None:
        pass

    def on_session_finished(self) -> None:
        pass


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
    def reporter(self) -> _RecordingCLIReporter:
        return _RecordingCLIReporter()

    @pytest.fixture
    def cli(self, reporter: _RecordingCLIReporter) -> SefiaCLI:
        return SefiaCLI(model="gpt-4o", reporter=reporter)

    def test_input_tool_is_exposed(self, cli: SefiaCLI):
        assert isinstance(cli.input_tool, Input)

    def test_output_tool_is_exposed(self, cli: SefiaCLI):
        assert isinstance(cli.output_tool, Output)

    async def test_create_session_becomes_active(
        self, cli: SefiaCLI, reporter: _RecordingCLIReporter
    ) -> None:
        session_id = cli.create_session()

        async with cli.session():
            pass

        assert reporter.resolved_sessions == [(session_id, "active")]

    async def test_switch_session(
        self, cli: SefiaCLI, reporter: _RecordingCLIReporter
    ) -> None:
        first = cli.create_session()
        cli.create_session()

        switched = cli.switch_session(first)
        async with cli.session():
            pass

        assert switched == first
        assert reporter.resolved_sessions == [(first, "active")]

    def test_switch_to_unknown_session_raises_cli_error(self, cli: SefiaCLI):
        # The facade translates the sefios-internal exception into the
        # sefia_typer one that applications catch.
        with pytest.raises(CLIUnknownSessionError) as exc_info:
            cli.switch_session("ghost")

        assert exc_info.value.session_id == "ghost"

    async def test_session_without_active_session_creates_and_reuses_one(
        self, cli: SefiaCLI, reporter: _RecordingCLIReporter
    ) -> None:
        async with cli.session():
            pass

        created_session_id, source = reporter.resolved_sessions[-1]
        assert source == "created"

        async with cli.session():
            pass

        assert reporter.resolved_sessions[-1] == (created_session_id, "active")

    async def test_registry_is_shared_but_active_selection_is_local(self) -> None:
        persistence = MemoryPersistence()
        first_reporter = _RecordingCLIReporter()
        second_reporter = _RecordingCLIReporter()
        first = SefiaCLI(
            model="gpt-4o",
            persistence=persistence,
            reporter=first_reporter,
        )
        second = SefiaCLI(
            model="gpt-4o",
            persistence=persistence,
            reporter=second_reporter,
        )

        first_session_id = first.create_session()

        async with second.session():
            pass

        second_session_id, source = second_reporter.resolved_sessions[-1]
        assert source == "created"
        assert second_session_id != first_session_id

        assert second.switch_session(first_session_id) == first_session_id
        async with second.session():
            pass

        assert second_reporter.resolved_sessions[-1] == (
            first_session_id,
            "active",
        )


async def test_non_interaction_pause_uses_generic_wording(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import typer
    from sefia.exceptions import PauseException

    class OtherPause(PauseException):
        pass

    cli = SefiaCLI(model="unused")
    with pytest.raises(typer.Exit) as exit_info:
        async with cli.session():
            raise OtherPause("later")
    assert exit_info.value.exit_code == 0
    output = capsys.readouterr().out
    assert "EXECUTION PAUSED" in output
    assert "input" not in output.lower()
    assert "INTERACTION_REQUIRED" not in output
