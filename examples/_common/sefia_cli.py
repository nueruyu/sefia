import inspect
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol, TypeVar, cast

import typer
from glyff.exceptions import YieldException
from sefia import Policy
from sefios import SessionScope
from sefios.tools import HumanInputRequest, HumanInputTool

from .human_input import CLIHumanInputAdapter, CLIHumanInputReceiver
from .policies import VerbosePolicy
from .session import ResolvedSession, SessionManager

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]
_USE_DEFAULT_REPORTER = object()
# Sentinel marking "no per-call override"; forwards the scope's configured
# max_steps instead of an explicit value (note: max_steps=None means no limit).
_USE_SCOPE_DEFAULT = object()


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

    def on_interrupted(
        self,
        session: ResolvedSession,
    ) -> MaybeAwaitable[None]: ...


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

    def on_interrupted(self, session: ResolvedSession) -> None:
        typer.echo()
        typer.secho("WAITING FOR INPUT", fg=typer.colors.YELLOW, bold=True)
        typer.echo("Session interrupted to wait for your input.")
        typer.echo("To resume, run the script again with your answer.")


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
        )
        self._human_input_tool = self._human_input.create_tool()
        self._human_input_receiver = CLIHumanInputReceiver(self._human_input.store)
        self._verbose = verbose

        self._session_scope = SessionScope(
            session_dir=session_dir,
            model=model,
            stream=stream,
            max_steps=max_steps,
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
        max_steps: int | None | object = _USE_SCOPE_DEFAULT,
    ) -> AsyncIterator[SefiaCLISession]:
        """Run code within a resolved Sefia CLI session context."""
        resolved_session = self._session_manager.resolve_session(session_id)

        resolved_verbose = self._verbose if verbose is None else verbose
        session_policies: list[Policy] | None = None
        if resolved_verbose:
            session_policies = [VerbosePolicy()]

        try:
            await self._report_session_resolved(resolved_session)
            if max_steps is _USE_SCOPE_DEFAULT:
                async with self._session_scope.session(
                    session_id=resolved_session.session_id,
                    model=model,
                    stream=stream,
                    policies=session_policies,
                ) as session:
                    with self._human_input.store.use_session_store(
                        session.session_store
                    ):
                        yield SefiaCLISession(human_input=self._human_input_receiver)
            else:
                async with self._session_scope.session(
                    session_id=resolved_session.session_id,
                    model=model,
                    stream=stream,
                    policies=session_policies,
                    max_steps=cast(int | None, max_steps),
                ) as session:
                    with self._human_input.store.use_session_store(
                        session.session_store
                    ):
                        yield SefiaCLISession(human_input=self._human_input_receiver)
        except YieldException:
            await self._report_interrupted(resolved_session)
            raise typer.Exit(code=0)

    async def _report_session_resolved(self, session: ResolvedSession) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_session_resolved(session))

    async def _report_human_input_request(self, request: HumanInputRequest) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_human_input_request(request))

    async def _report_interrupted(self, session: ResolvedSession) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_interrupted(session))

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
