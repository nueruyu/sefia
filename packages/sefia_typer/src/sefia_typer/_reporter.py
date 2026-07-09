from typing import Protocol

import typer
from sefia.exceptions import InferenceError

from ._input import InputRequest, MaybeAwaitable


class ResolvedSessionLike(Protocol):
    """The session-resolution facts a reporter renders.

    Read-only by design: any object with these attributes (such as the
    integration layer's resolved-session type) satisfies it structurally.
    """

    @property
    def session_id(self) -> str: ...

    @property
    def source(self) -> str: ...


class CLIReporter(Protocol):
    """Receives CLI lifecycle events and renders them for the host application."""

    def on_session_resolved(
        self,
        session: ResolvedSessionLike,
    ) -> MaybeAwaitable[None]: ...

    def on_input_request(
        self,
        request: InputRequest,
    ) -> MaybeAwaitable[None]: ...

    def on_input_prompt_delta(self, text: str) -> MaybeAwaitable[None]: ...

    def on_interrupted(
        self,
        session: ResolvedSessionLike,
    ) -> MaybeAwaitable[None]: ...

    def on_inference_error(self, error: InferenceError) -> MaybeAwaitable[None]: ...

    def on_session_finished(self) -> MaybeAwaitable[None]: ...


class DefaultCLIReporter(CLIReporter):
    """Default CLI reporter using Typer's standard terminal output helpers."""

    def on_session_resolved(self, session: ResolvedSessionLike) -> None:
        if session.source == "created":
            typer.secho(
                f"> No active session. Starting new session: {session.session_id}",
                bold=True,
            )
        elif session.source == "active":
            typer.secho(f"> Resuming session {session.session_id}", bold=True)

    def on_input_request(self, request: InputRequest) -> None:
        typer.echo()
        typer.secho(
            f"[INPUT_REQUIRED:{request.interaction_id}]",
            fg=typer.colors.YELLOW,
            bold=True,
            nl=False,
        )
        typer.echo(f" {request.prompt}")
        typer.echo()

    def on_input_prompt_delta(self, text: str) -> None:
        typer.echo(text, nl=False)

    def on_interrupted(self, session: ResolvedSessionLike) -> None:
        typer.echo()
        typer.secho("WAITING FOR INPUT", fg=typer.colors.YELLOW, bold=True)
        typer.echo("Session interrupted to wait for your input.")
        typer.echo("To resume, run the script again with your input.")

    def on_inference_error(self, error: InferenceError) -> None:
        typer.echo()
        typer.secho("INFERENCE ERROR", fg=typer.colors.RED, bold=True)
        typer.echo(str(error))

    def on_session_finished(self) -> None:
        pass
