import pytest
import typer
from glyff.exceptions import YieldException
from pytest_mock import MockerFixture

from examples._common.sefia_cli import SefiaCLI


class TestCostReporting:
    async def test_reports_on_normal_completion(self, tmp_path, mocker: MockerFixture):
        reporter = mocker.Mock()
        cli = SefiaCLI(
            session_dir=tmp_path / "s",
            model="gpt-4o",
            stream=False,
            reporter=reporter,
        )

        async with cli.session():
            pass

        reporter.on_session_finished.assert_called_once()

    async def test_reports_on_yield(self, tmp_path, mocker: MockerFixture):
        # A human-input interrupt (chat-style loop) raises YieldException from
        # inside the session block; cost should still be reported at that point.
        reporter = mocker.Mock()
        cli = SefiaCLI(
            session_dir=tmp_path / "s",
            model="gpt-4o",
            stream=False,
            reporter=reporter,
        )

        with pytest.raises(typer.Exit):
            async with cli.session():
                raise YieldException()

        reporter.on_session_finished.assert_called_once()
        reporter.on_interrupted.assert_called_once()
