from unittest.mock import Mock

from sefia_typer import CLIReporter, OutputMessage as CLIOutputMessage
from sefios.cli._reporting import CLIReporting
from sefios.tools import OutputMessage


async def test_output_is_converted_to_reporter_message():
    reporter = Mock(spec=CLIReporter)
    reporting = CLIReporting(reporter)

    await reporting.output(OutputMessage(interaction_id="call-1", message="hello"))

    reporter.on_output.assert_called_once_with(
        CLIOutputMessage(interaction_id="call-1", message="hello")
    )


async def test_disabled_reporting_is_a_noop():
    reporting = CLIReporting(None)

    await reporting.output(OutputMessage(interaction_id="call-1", message="hello"))
    await reporting.session_finished()
