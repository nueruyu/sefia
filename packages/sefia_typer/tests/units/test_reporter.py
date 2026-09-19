import json
from dataclasses import dataclass

import pytest
from sefia.exceptions import InvalidInferenceResponseError
from sefia_typer import DefaultCLIReporter, InteractionRequest, OutputMessage


@dataclass(frozen=True)
class _StubResolvedSession:
    session_id: str
    source: str


@dataclass(frozen=True)
class _StubInteractionRequest:
    interaction_id: str
    payload: object


class TestDefaultCLIReporter:
    def test_created_session_is_announced(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reporter = DefaultCLIReporter()

        reporter.on_session_resolved(
            _StubResolvedSession(session_id="abc", source="created")
        )

        assert "abc" in capsys.readouterr().out

    def test_active_session_is_announced(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reporter = DefaultCLIReporter()

        reporter.on_session_resolved(
            _StubResolvedSession(session_id="abc", source="active")
        )

        assert "abc" in capsys.readouterr().out

    def test_explicit_session_is_quiet(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reporter = DefaultCLIReporter()

        reporter.on_session_resolved(
            _StubResolvedSession(session_id="abc", source="explicit")
        )

        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize(
        "payload",
        [
            {"type": "weather", "arguments": {"city": "Tokyo"}},
            {"type": "input", "prompt": "Continue?"},
            ["opaque", None],
        ],
    )
    def test_interaction_request_renders_opaque_json(
        self, capsys: pytest.CaptureFixture[str], payload: object
    ) -> None:
        reporter = DefaultCLIReporter()
        request: InteractionRequest = _StubInteractionRequest("xyz", payload)
        reporter.on_interaction_request(request)
        marker, rendered = capsys.readouterr().out.strip().split(" ", 1)
        assert marker == "[INTERACTION_REQUIRED:xyz]"
        assert json.loads(rendered) == payload

    def test_input_prompt_delta_is_printed_without_newline(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reporter = DefaultCLIReporter()

        reporter.on_input_prompt_delta("call-1", "What ")
        reporter.on_input_prompt_delta("call-1", "topic?")

        assert capsys.readouterr().out == "What topic?"

    def test_output_includes_marker_and_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reporter = DefaultCLIReporter()

        reporter.on_output(OutputMessage(interaction_id="xyz", message="Hello there!"))

        output = capsys.readouterr().out
        assert "OUTPUT:xyz" in output
        assert "Hello there!" in output

    def test_output_message_delta_is_printed_without_newline(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reporter = DefaultCLIReporter()

        reporter.on_output_message_delta("call-1", "Hello ")
        reporter.on_output_message_delta("call-1", "there!")

        assert capsys.readouterr().out == "Hello there!"

    def test_interrupted_announces_generic_pause(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reporter = DefaultCLIReporter()

        reporter.on_interrupted(_StubResolvedSession(session_id="abc", source="active"))

        output = capsys.readouterr().out
        assert "EXECUTION PAUSED" in output
        assert "resumed later" in output
        assert "input" not in output.lower()
        assert "interaction" not in output.lower()

    def test_inference_error_is_reported_as_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reporter = DefaultCLIReporter()

        reporter.on_inference_error(InvalidInferenceResponseError("bad model response"))

        output = capsys.readouterr().out
        assert "INFERENCE ERROR" in output
        assert "bad model response" in output
        assert "EXECUTION PAUSED" not in output
