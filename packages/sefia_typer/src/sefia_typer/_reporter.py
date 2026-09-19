import json
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol, TypeVar

import typer
from sefia.exceptions import InferenceError
from typing_extensions import final, override

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]


class InteractionRequest(Protocol):
    """An opaque JSON-compatible request supplied by an integration."""

    @property
    def interaction_id(self) -> str: ...

    @property
    def payload(self) -> object: ...


@dataclass(frozen=True)
class OutputMessage:
    """A message the agent emitted to the human without waiting for a reply."""

    interaction_id: str
    message: str


class ResolvedSession(Protocol):
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
        session: ResolvedSession,
    ) -> MaybeAwaitable[None]: ...

    def on_interaction_request(
        self,
        request: InteractionRequest,
    ) -> MaybeAwaitable[None]: ...

    def on_input_prompt_delta(
        self, interaction_id: str, text: str
    ) -> MaybeAwaitable[None]: ...

    def on_output(self, message: OutputMessage) -> MaybeAwaitable[None]: ...

    def on_output_message_delta(
        self, interaction_id: str, text: str
    ) -> MaybeAwaitable[None]: ...

    def on_interrupted(
        self,
        session: ResolvedSession,
    ) -> MaybeAwaitable[None]: ...

    def on_inference_error(self, error: InferenceError) -> MaybeAwaitable[None]: ...

    def on_session_finished(self) -> MaybeAwaitable[None]: ...


@final
class DefaultCLIReporter(CLIReporter):
    """Default CLI reporter using Typer's standard terminal output helpers."""

    @override
    def on_session_resolved(self, session: ResolvedSession) -> None:
        if session.source == "created":
            typer.secho(
                f"> No active session. Starting new session: {session.session_id}",
                bold=True,
            )
        elif session.source == "active":
            typer.secho(f"> Resuming session {session.session_id}", bold=True)

    @override
    def on_interaction_request(self, request: InteractionRequest) -> None:
        typer.echo()
        typer.secho(
            f"[INTERACTION_REQUIRED:{request.interaction_id}]",
            fg=typer.colors.YELLOW,
            bold=True,
            nl=False,
        )
        typer.echo(f" {json.dumps(request.payload, ensure_ascii=False)}")
        typer.echo()

    @override
    def on_input_prompt_delta(self, interaction_id: str, text: str) -> None:
        typer.echo(text, nl=False)

    @override
    def on_output(self, message: OutputMessage) -> None:
        typer.echo()
        typer.secho(
            f"[OUTPUT:{message.interaction_id}]",
            fg=typer.colors.CYAN,
            bold=True,
            nl=False,
        )
        typer.echo(f" {message.message}")
        typer.echo()

    @override
    def on_output_message_delta(self, interaction_id: str, text: str) -> None:
        typer.echo(text, nl=False)

    @override
    def on_interrupted(self, session: ResolvedSession) -> None:
        typer.echo()
        typer.secho("EXECUTION PAUSED", fg=typer.colors.YELLOW, bold=True)
        typer.echo("The session was interrupted and can be resumed later.")

    @override
    def on_inference_error(self, error: InferenceError) -> None:
        typer.echo()
        typer.secho("INFERENCE ERROR", fg=typer.colors.RED, bold=True)
        typer.echo(str(error))

    @override
    def on_session_finished(self) -> None:
        pass
