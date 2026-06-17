import asyncio
import functools
import inspect
from collections.abc import Awaitable, Callable
from enum import Enum, auto
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Protocol,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import typer
from glyff.exceptions import YieldException
from sefios import SefiaScope
from sefios.tools import HumanInputRequest, HumanInputTool

from .human_input import CLIHumanInputAdapter
from .session import ResolvedSession, SessionManager

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]
_USE_DEFAULT_REPORTER = object()
_DEFAULT_SESSION_ID_PARAM = "session_id"
_DEFAULT_MODEL_PARAM = "model"
_DEFAULT_VERBOSE_PARAM = "verbose"


class CLIParam(Enum):
    """Marks command parameters consumed by SefiaCLI."""

    SESSION_ID = auto()
    MODEL = auto()
    VERBOSE = auto()
    INPUT = auto()
    REPLY_TO = auto()


class CLIReporter(Protocol):
    """Receives CLI lifecycle events and renders them for the host application."""

    def on_session_resolved(
        self,
        session: ResolvedSession,
    ) -> MaybeAwaitable[None]:
        ...

    def on_human_input_request(
        self,
        request: HumanInputRequest,
    ) -> MaybeAwaitable[None]:
        ...

    def on_interrupted(
        self,
        session: ResolvedSession,
    ) -> MaybeAwaitable[None]:
        ...


class DefaultCLIReporter:
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

    def on_interrupted(self, _session: ResolvedSession) -> None:
        typer.echo()
        typer.secho("WAITING FOR INPUT", fg=typer.colors.YELLOW, bold=True)
        typer.echo("Session interrupted to wait for your input.")
        typer.echo("To resume, run the script again with your answer.")


class SefiaCLI:
    """Runs Typer command callbacks inside a Sefia session context."""

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

        self._sefia_scope = SefiaScope(
            session_dir=session_dir,
            model=model,
            stream=stream,
            verbose=verbose,
            max_steps=max_steps,
        )
        self._scoped_run = self._sefia_scope(self._run_scoped_command)

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

    def scope(self, func: Callable[..., T] | None = None):
        """Decorate a Typer command callback to run inside a Sefia CLI scope."""

        def decorator(inner: Callable[..., T]) -> Callable[..., Any]:
            signature = inspect.signature(inner)
            cli_param_names = _find_cli_param_names(inner)
            session_id_param = cli_param_names.get(
                CLIParam.SESSION_ID,
                _DEFAULT_SESSION_ID_PARAM,
            )
            model_param = cli_param_names.get(CLIParam.MODEL, _DEFAULT_MODEL_PARAM)
            verbose_param = cli_param_names.get(CLIParam.VERBOSE, _DEFAULT_VERBOSE_PARAM)

            @functools.wraps(inner)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()

                session_id = bound.arguments.get(session_id_param)
                resolved_session = self._session_manager.resolve_session(session_id)

                run_kwargs: dict[str, Any] = {
                    "session_id": resolved_session.session_id,
                    "func": inner,
                    "bound": bound,
                    "cli_param_names": cli_param_names,
                    "resolved_session": resolved_session,
                }
                if model_param in bound.arguments:
                    run_kwargs["model"] = bound.arguments[model_param]
                if verbose_param in bound.arguments:
                    run_kwargs["verbose"] = bound.arguments[verbose_param]

                return asyncio.run(self._scoped_run(**run_kwargs))

            wrapper.__signature__ = signature  # type: ignore[attr-defined]
            return wrapper

        if func is not None:
            return decorator(func)

        return decorator

    async def _run_scoped_command(
        self,
        *,
        func: Callable[..., Any],
        bound: inspect.BoundArguments,
        cli_param_names: dict[CLIParam, str],
        resolved_session: ResolvedSession,
    ) -> None:
        try:
            await self._report_session_resolved(resolved_session)
            await self._accept_cli_input(bound, cli_param_names)

            result = func(*bound.args, **bound.kwargs)
            if inspect.isawaitable(result):
                await result

        except YieldException:
            await self._report_interrupted(resolved_session)
            raise typer.Exit(code=0)

    async def _accept_cli_input(
        self,
        bound: inspect.BoundArguments,
        cli_param_names: dict[CLIParam, str],
    ) -> None:
        input_param = cli_param_names.get(CLIParam.INPUT)
        if input_param is None:
            return

        input_value = bound.arguments[input_param]
        if input_value is None:
            return

        input_text = (
            " ".join(input_value).strip()
            if isinstance(input_value, list)
            else str(input_value).strip()
        )
        reply_to = None
        reply_to_param = cli_param_names.get(CLIParam.REPLY_TO)
        if reply_to_param is not None:
            reply_to = bound.arguments.get(reply_to_param)

        await self._human_input.receive_input(input_text, reply_to=reply_to)

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


def _find_cli_param_names(func: Callable[..., Any]) -> dict[CLIParam, str]:
    hints = get_type_hints(func, include_extras=True)
    result: dict[CLIParam, str] = {}

    for name, hint in hints.items():
        if get_origin(hint) is not Annotated:
            continue

        for metadata in get_args(hint)[1:]:
            if not isinstance(metadata, CLIParam):
                continue

            if metadata in result:
                raise TypeError(
                    f"Duplicate Sefia CLI parameter marker: {metadata.name}."
                )
            result[metadata] = name

    return result


async def _maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value
