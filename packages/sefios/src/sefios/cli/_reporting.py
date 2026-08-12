from sefia.exceptions import InferenceError
from sefia_typer import CLIReporter
from sefia_typer import InputRequest as CLIInputRequest
from sefia_typer import OutputMessage as CLIOutputMessage

from .._async import maybe_await
from ..sessions import ResolvedSession
from ..tools import OutputMessage


class CLIReporting:
    """Bridge sefios session/tool events to a sefia-typer reporter."""

    def __init__(self, reporter: CLIReporter | None):
        self.reporter = reporter

    async def session_resolved(self, session: ResolvedSession) -> None:
        if self.reporter is not None:
            await maybe_await(self.reporter.on_session_resolved(session))

    async def input_request(self, request: CLIInputRequest) -> None:
        if self.reporter is not None:
            await maybe_await(self.reporter.on_input_request(request))

    async def input_prompt_delta(self, interaction_id: str, text: str) -> None:
        if self.reporter is not None:
            await maybe_await(self.reporter.on_input_prompt_delta(interaction_id, text))

    async def output(self, message: OutputMessage) -> None:
        if self.reporter is not None:
            await maybe_await(
                self.reporter.on_output(
                    CLIOutputMessage(
                        interaction_id=message.interaction_id,
                        message=message.message,
                    )
                )
            )

    async def output_message_delta(self, interaction_id: str, text: str) -> None:
        if self.reporter is not None:
            await maybe_await(
                self.reporter.on_output_message_delta(interaction_id, text)
            )

    async def interrupted(self, session: ResolvedSession) -> None:
        if self.reporter is not None:
            await maybe_await(self.reporter.on_interrupted(session))

    async def inference_error(self, error: InferenceError) -> None:
        if self.reporter is not None:
            await maybe_await(self.reporter.on_inference_error(error))

    async def session_finished(self) -> None:
        if self.reporter is not None:
            await maybe_await(self.reporter.on_session_finished())
