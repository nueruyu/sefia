import inspect
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol, TypeVar, cast

import typer
from sefia import Policy
from sefia.exceptions import InferenceError, PauseException
from sefios import SessionScope, get_session_storage, get_state
from sefios.handlers import CostCalculator, CostState
from sefios.tools import HumanInputRequest, HumanInputTool

from .human_input import CLIHumanInputAdapter, CLIHumanInputReceiver
from .policies import VerbosePolicy
from .session import ResolvedSession, SessionManager

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]
_USE_DEFAULT_REPORTER = object()


class CLIReporter(Protocol):
    """Receives CLI lifecycle events and renders them for the host application."""

    def on_session_resolved(
        self,
        session: ResolvedSession,
    ) -> MaybeAwaitable[None]: ...

    def on_human_input_request(
        self,
        request: HumanInputRequest,
    ) -> MaybeAwaitable[None]: ...

    def on_human_input_question_delta(self, text: str) -> MaybeAwaitable[None]: ...

    def on_interrupted(
        self,
        session: ResolvedSession,
    ) -> MaybeAwaitable[None]: ...

    def on_inference_error(self, error: InferenceError) -> MaybeAwaitable[None]: ...

    def on_session_finished(self) -> MaybeAwaitable[None]: ...


class DefaultCLIReporter(CLIReporter):
    """Default CLI reporter using Typer's standard terminal output helpers."""

    def on_session_resolved(self, session: ResolvedSession) -> None:
        if session.source == "created":
            typer.secho(
                f"> No active session. Starting new session: {session.session_id}",
                bold=True,
            )
        elif session.source == "active":
            typer.secho(f"> Resuming session {session.session_id}", bold=True)

    def on_human_input_request(self, request: HumanInputRequest) -> None:
        typer.echo()
        typer.secho(
            f"[USER_INPUT_REQUIRED:{request.interaction_id}]",
            fg=typer.colors.YELLOW,
            bold=True,
            nl=False,
        )
        typer.echo(f" {request.question}")
        typer.echo()

    def on_human_input_question_delta(self, text: str) -> None:
        typer.echo(text, nl=False)

    async def on_interrupted(self, session: ResolvedSession) -> None:
        typer.echo()
        typer.secho("WAITING FOR INPUT", fg=typer.colors.YELLOW, bold=True)
        typer.echo("Session interrupted to wait for your input.")
        typer.echo("To resume, run the script again with your answer.")
        await self._echo_total_cost()

    async def on_inference_error(self, error: InferenceError) -> None:
        typer.echo()
        typer.secho("INFERENCE ERROR", fg=typer.colors.RED, bold=True)
        typer.echo(str(error))
        await self._echo_total_cost()

    async def on_session_finished(self) -> None:
        await self._echo_total_cost()

    @staticmethod
    async def _echo_total_cost() -> None:
        cost_state = await get_state().get(CostState).ensure()
        typer.echo()
        typer.secho(f"> Total cost: ${cost_state.cost:.4f}", bold=True)


class SefiaCLISession:
    """Operations available inside a Sefia CLI session context."""

    def __init__(self, *, human_input: CLIHumanInputReceiver):
        self._human_input = human_input

    async def accept_input(
        self,
        input_value: str | list[str] | None,
        *,
        reply_to: str | None = None,
    ) -> None:
        """Store CLI input as an answer for a pending or upcoming interaction."""
        if input_value is None:
            return

        input_text = _to_input_text(input_value)
        if not input_text:
            return

        await self._human_input.receive_input(input_text, reply_to=reply_to)


class SefiaCLI:
    """Creates Sefia session contexts for Typer commands."""

    def __init__(
        self,
        *,
        session_dir: Path,
        human_input_adapter: CLIHumanInputAdapter | None = None,
        reporter: CLIReporter | None | object = _USE_DEFAULT_REPORTER,
        model: str | None = None,
        stream: bool = True,
        verbose: bool = False,
        max_steps: int | None = 25,
    ):
        self._reporter = self._resolve_reporter(reporter)
        self._session_manager = SessionManager(session_dir)
        self._human_input = human_input_adapter or CLIHumanInputAdapter(
            on_request=self._report_human_input_request,
            on_question_delta=self._report_human_input_question_delta,
        )
        self._human_input_tool = self._human_input.create_tool()
        self._human_input_receiver = CLIHumanInputReceiver(self._human_input.store)
        self._verbose = verbose

        scope_policies: list[Policy] = [
            Policy(handlers=lambda: [CostCalculator()])
        ]
        self._session_scope = SessionScope(
            session_dir=session_dir,
            model=model,
            stream=stream,
            max_steps=max_steps,
            policies=scope_policies,
        )

    @property
    def human_input_tool(self) -> HumanInputTool:
        return self._human_input_tool

    def create_session(self) -> str:
        """Create a new active CLI session and return its ID."""
        return self._session_manager.create_new_active_session()

    def switch_session(self, session_id: str) -> str:
        """Switch the active CLI session and return its ID."""
        return self._session_manager.switch_active_session(session_id)

    def get_active_session(self) -> str | None:
        """Return the active CLI session ID, if any."""
        return self._session_manager.get_active_session_id()

    @asynccontextmanager
    async def session(
        self,
        *,
        session_id: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        verbose: bool | None = None,
    ) -> AsyncIterator[SefiaCLISession]:
        """Run code within a resolved Sefia CLI session context."""
        resolved_session = self._session_manager.resolve_session(session_id)

        resolved_verbose = self._verbose if verbose is None else verbose
        session_policies: list[Policy] | None = None
        if resolved_verbose:
            session_policies = [VerbosePolicy()]

        try:
            await self._report_session_resolved(resolved_session)
            async with self._session_scope.session(
                session_id=resolved_session.session_id,
                model=model,
                stream=stream,
                policies=session_policies,
            ):
                with self._human_input.store.use_session_storage(get_session_storage()):
                    try:
                        yield SefiaCLISession(human_input=self._human_input_receiver)
                    except InferenceError as e:
                        await self._report_inference_error(e)
                        raise
                    except PauseException:
                        # Any pause (NeedsInput, or a future pause type) is a
                        # graceful interrupt, not a failure. The session context
                        # is still alive here, so reporters may read running
                        # state (e.g. cost) via get_state().
                        await self._report_interrupted(resolved_session)
                        raise
                    else:
                        await self._report_session_finished()
        except InferenceError:
            raise typer.Exit(code=1) from None
        except PauseException:
            raise typer.Exit(code=0)

    async def _report_session_resolved(self, session: ResolvedSession) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_session_resolved(session))

    async def _report_human_input_request(self, request: HumanInputRequest) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_human_input_request(request))

    async def _report_human_input_question_delta(self, text: str) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_human_input_question_delta(text))

    async def _report_interrupted(self, session: ResolvedSession) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_interrupted(session))

    async def _report_inference_error(self, error: InferenceError) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_inference_error(error))

    async def _report_session_finished(self) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_session_finished())

    @staticmethod
    def _resolve_reporter(reporter: CLIReporter | None | object) -> CLIReporter | None:
        if reporter is _USE_DEFAULT_REPORTER:
            return DefaultCLIReporter()
        return cast(CLIReporter | None, reporter)


def _to_input_text(input_value: str | list[str]) -> str:
    if isinstance(input_value, str):
        return input_value.strip()
    return " ".join(input_value).strip()


async def _maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value
