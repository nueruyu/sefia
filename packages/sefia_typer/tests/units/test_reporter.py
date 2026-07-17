from dataclasses import dataclass

from sefia.exceptions import InvalidInferenceResponseError
from sefia_typer import DefaultCLIReporter, InputRequest, OutputMessage


@dataclass(frozen=True)
class _StubResolvedSession:
    session_id: str
    source: str


class TestDefaultCLIReporter:
    def test_created_session_is_announced(self, capsys):
        reporter = DefaultCLIReporter()

        reporter.on_session_resolved(
            _StubResolvedSession(session_id="abc", source="created")
        )

        assert "abc" in capsys.readouterr().out

    def test_active_session_is_announced(self, capsys):
        reporter = DefaultCLIReporter()

        reporter.on_session_resolved(
            _StubResolvedSession(session_id="abc", source="active")
        )

        assert "abc" in capsys.readouterr().out

    def test_explicit_session_is_quiet(self, capsys):
        reporter = DefaultCLIReporter()

        reporter.on_session_resolved(
            _StubResolvedSession(session_id="abc", source="explicit")
        )

        assert capsys.readouterr().out == ""

    def test_input_request_includes_marker(self, capsys):
        reporter = DefaultCLIReporter()

        reporter.on_input_request(
            InputRequest(interaction_id="xyz", prompt="What topic?")
        )

        output = capsys.readouterr().out
        assert "INPUT_REQUIRED:xyz" in output
        assert "What topic?" in output

    def test_input_prompt_delta_is_printed_without_newline(self, capsys):
        reporter = DefaultCLIReporter()

        reporter.on_input_prompt_delta("What ")
        reporter.on_input_prompt_delta("topic?")

        assert capsys.readouterr().out == "What topic?"

    def test_output_includes_marker_and_message(self, capsys):
        reporter = DefaultCLIReporter()

        reporter.on_output(OutputMessage(interaction_id="xyz", message="Hello there!"))

        output = capsys.readouterr().out
        assert "OUTPUT:xyz" in output
        assert "Hello there!" in output

    def test_output_message_delta_is_printed_without_newline(self, capsys):
        reporter = DefaultCLIReporter()

        reporter.on_output_message_delta("Hello ")
        reporter.on_output_message_delta("there!")

        assert capsys.readouterr().out == "Hello there!"

    def test_interrupted_announces_waiting_state(self, capsys):
        reporter = DefaultCLIReporter()

        reporter.on_interrupted(_StubResolvedSession(session_id="abc", source="active"))

        assert "WAITING FOR INPUT" in capsys.readouterr().out

    def test_inference_error_is_reported_as_error(self, capsys):
        reporter = DefaultCLIReporter()

        reporter.on_inference_error(InvalidInferenceResponseError("bad model response"))

        output = capsys.readouterr().out
        assert "INFERENCE ERROR" in output
        assert "bad model response" in output
        assert "WAITING FOR INPUT" not in output
