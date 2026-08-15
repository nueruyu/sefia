import pytest
import typer
from pytest_mock import MockerFixture
from sefia.exceptions import InvalidInferenceResponseError
from sefios.cli import SefiaCLI
from sefios.exceptions import InputRequired


class TestCostReporting:
    async def test_reports_on_normal_completion(self, mocker: MockerFixture):
        reporter = mocker.Mock()
        cli = SefiaCLI(
            model="gpt-4o",
            stream=False,
            reporter=reporter,
        )

        async with cli.session():
            pass

        reporter.on_session_finished.assert_called_once()

    async def test_reports_on_yield(self, mocker: MockerFixture):
        # A input interrupt (chat-style loop) raises InputRequired from
        # inside the session block; cost should still be reported at that point.
        reporter = mocker.Mock()
        cli = SefiaCLI(
            model="gpt-4o",
            stream=False,
            reporter=reporter,
        )

        with pytest.raises(typer.Exit):
            async with cli.session():
                raise InputRequired("What is your name?")

        # On a yield, the interrupt hook fires (reporters may read running cost
        # via get_state there) but the session did not finish normally.
        reporter.on_interrupted.assert_called_once()
        reporter.on_session_finished.assert_not_called()

    async def test_inference_error_is_not_reported_as_input_wait(
        self, mocker: MockerFixture
    ):
        reporter = mocker.Mock()
        cli = SefiaCLI(
            model="gpt-4o",
            stream=False,
            reporter=reporter,
        )

        with pytest.raises(typer.Exit) as exc_info:
            async with cli.session():
                raise InvalidInferenceResponseError("bad model response")

        assert exc_info.value.exit_code == 1
        reporter.on_inference_error.assert_called_once()
        reporter.on_interrupted.assert_not_called()
        reporter.on_session_finished.assert_not_called()
