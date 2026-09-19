import typer
from sefia.exceptions import InferenceError
from sefia_typer import CLIReporter, DefaultCLIReporter, InputRequest, OutputMessage
from sefia_typer import ResolvedSession as CLIResolvedSession
from typing_extensions import final, override

from .._async import MaybeAwaitable, maybe_await
from ..handlers import CostState
from ..state import get_state


@final
class CostReportingCLIReporter(CLIReporter):
    """Add session cost output to another CLI reporter."""

    def __init__(self, inner: CLIReporter | None = None):
        self._inner = inner or DefaultCLIReporter()

    @override
    def on_session_resolved(self, session: CLIResolvedSession) -> MaybeAwaitable[None]:
        return self._inner.on_session_resolved(session)

    @override
    def on_input_request(self, request: InputRequest) -> MaybeAwaitable[None]:
        return self._inner.on_input_request(request)

    @override
    def on_input_prompt_delta(
        self, interaction_id: str, text: str
    ) -> MaybeAwaitable[None]:
        return self._inner.on_input_prompt_delta(interaction_id, text)

    @override
    def on_output(self, message: OutputMessage) -> MaybeAwaitable[None]:
        return self._inner.on_output(message)

    @override
    def on_output_message_delta(
        self, interaction_id: str, text: str
    ) -> MaybeAwaitable[None]:
        return self._inner.on_output_message_delta(interaction_id, text)

    @override
    async def on_interrupted(self, session: CLIResolvedSession) -> None:
        await maybe_await(self._inner.on_interrupted(session))
        await self._echo_total_cost()

    @override
    async def on_inference_error(self, error: InferenceError) -> None:
        await maybe_await(self._inner.on_inference_error(error))
        await self._echo_total_cost()

    @override
    async def on_session_finished(self) -> None:
        await maybe_await(self._inner.on_session_finished())
        await self._echo_total_cost()

    @staticmethod
    async def _echo_total_cost() -> None:
        cost_state = await get_state().get(CostState).ensure()
        typer.echo()
        typer.secho(f"> Total cost: ${cost_state.cost:.4f}", bold=True)
