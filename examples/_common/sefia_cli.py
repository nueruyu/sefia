import asyncio
import contextvars
import functools
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import typer
from glyff.exceptions import YieldException
from sefia import get_context
from sefios import SefiaScope
from sefios.tools import HumanInputTool

from .human_input import CLIHumanInputAdapter
from .session import ResolvedSession, SessionManager, SessionSource
from .workflow import WorkflowState

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]


@dataclass(frozen=True)
class CLISessionState:
    """Input and session state for a Sefia CLI command invocation."""

    session_id: str
    source: SessionSource
    is_new_session: bool
    current_input: str
    initial_input: str


@dataclass(frozen=True)
class SefiaCLIEvents:
    """Callbacks for UI or host-specific CLI behavior."""

    on_session_resolved: Callable[[ResolvedSession], MaybeAwaitable[None]] | None = None
    on_interrupted: Callable[[CLISessionState | None], MaybeAwaitable[None]] | None = None


@dataclass(frozen=True)
class _CLIInvocation:
    """Internal runtime information for the currently running CLI command."""

    resolved_session: ResolvedSession
    session_state: CLISessionState | None = None

    @property
    def input_accepted(self) -> bool:
        return self.session_state is not None


class SefiaCLI:
    """Runs Typer command callbacks inside a Sefia session context."""

    def __init__(
        self,
        *,
        session_dir: Path,
        human_input_adapter: CLIHumanInputAdapter | None = None,
        events: SefiaCLIEvents | None = None,
        model: str | None = None,
        stream: bool = True,
        verbose: bool = False,
        max_steps: int | None = 25,
    ):
        self._session_manager = SessionManager(session_dir)
        self._human_input = human_input_adapter or CLIHumanInputAdapter()
        self._human_input_tool = self._human_input.create_tool()
        self._events = events or SefiaCLIEvents()

        self._sefia_scope = SefiaScope(
            session_dir=session_dir,
            model=model,
            stream=stream,
            verbose=verbose,
            max_steps=max_steps,
        )
        self._scoped_run = self._sefia_scope(self._run_scoped_command)
        self._invocation_var: contextvars.ContextVar[_CLIInvocation | None] = (
            contextvars.ContextVar("sefia_cli_invocation", default=None)
        )

    @property
    def human_input_tool(self) -> HumanInputTool:
        return self._human_input_tool

    async def accept_input(self, input_value: str | list[str]) -> CLISessionState:
        """
        Accept the current CLI input and return the current CLI session state.

        New or uninitialized sessions store the input as the workflow initial input.
        Resumed sessions forward the input to a pending human interaction when present.
        """
        invocation = self._get_invocation()
        if invocation.input_accepted:
            raise RuntimeError("Input has already been accepted for this invocation.")

        input_text = self._to_input_text(input_value)
        await self._human_input.receive_input(
            input_text,
            is_new=invocation.resolved_session.is_new,
        )

        workflow_state = await self._get_workflow_state()
        if not workflow_state.has_initial_input:
            workflow_state = await self._initialize_workflow(input_text)

        session_state = CLISessionState(
            session_id=invocation.resolved_session.session_id,
            source=invocation.resolved_session.source,
            is_new_session=invocation.resolved_session.is_new,
            current_input=input_text,
            initial_input=workflow_state.require_initial_input(),
        )
        self._invocation_var.set(
            _CLIInvocation(
                resolved_session=invocation.resolved_session,
                session_state=session_state,
            )
        )
        return session_state

    def create_session(self) -> str:
        """Create a new active CLI session and return its ID."""
        return self._session_manager.create_new_active_session()

    def switch_session(self, session_id: str) -> str:
        """Switch the active CLI session and return its ID."""
        return self._session_manager.switch_active_session(session_id)

    def get_active_session(self) -> str | None:
        """Return the active CLI session ID, if any."""
        return self._session_manager.get_active_session_id()

    def scope(
        self,
        func: Callable[..., T] | None = None,
        *,
        session_id_arg: str = "session_id",
        model_arg: str = "model",
        verbose_arg: str = "verbose",
    ):
        """Decorate a Typer command callback to run inside a Sefia CLI scope."""

        def decorator(inner: Callable[..., T]) -> Callable[..., Any]:
            signature = inspect.signature(inner)

            @functools.wraps(inner)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()

                session_id = bound.arguments.get(session_id_arg)
                resolved_session = self._session_manager.resolve_session(session_id)

                run_kwargs: dict[str, Any] = {
                    "session_id": resolved_session.session_id,
                    "func": inner,
                    "bound": bound,
                    "resolved_session": resolved_session,
                }
                if model_arg in bound.arguments:
                    run_kwargs["model"] = bound.arguments[model_arg]
                if verbose_arg in bound.arguments:
                    run_kwargs["verbose"] = bound.arguments[verbose_arg]

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
        resolved_session: ResolvedSession,
    ) -> None:
        invocation = _CLIInvocation(resolved_session=resolved_session)
        token = self._invocation_var.set(invocation)

        try:
            await self._emit_session_resolved(resolved_session)

            result = func(*bound.args, **bound.kwargs)
            if inspect.isawaitable(result):
                await result

            invocation = self._get_invocation()
            if not invocation.input_accepted:
                raise RuntimeError(
                    "Sefia CLI command must call 'await sefia_cli.accept_input(...)'."
                )

        except YieldException:
            invocation = self._invocation_var.get()
            await self._emit_interrupted(
                invocation.session_state if invocation is not None else None
            )
            raise typer.Exit(code=0)
        finally:
            self._invocation_var.reset(token)

    async def _initialize_workflow(self, initial_input: str) -> WorkflowState:
        session = get_context()
        state_store = session.get_state_store("workflow_state", WorkflowState)
        state = WorkflowState.from_initial_input(initial_input)
        await state_store.save(state)
        return state

    async def _get_workflow_state(self) -> WorkflowState:
        session = get_context()
        state_store = session.get_state_store("workflow_state", WorkflowState)
        return await state_store.ensure()

    def _get_invocation(self) -> _CLIInvocation:
        invocation = self._invocation_var.get()
        if invocation is None:
            raise RuntimeError("No active Sefia CLI invocation.")
        return invocation

    async def _emit_session_resolved(self, session: ResolvedSession) -> None:
        if self._events.on_session_resolved is not None:
            await _maybe_await(self._events.on_session_resolved(session))

    async def _emit_interrupted(self, state: CLISessionState | None) -> None:
        if self._events.on_interrupted is not None:
            await _maybe_await(self._events.on_interrupted(state))

    @staticmethod
    def _to_input_text(input_value: str | list[str]) -> str:
        if isinstance(input_value, str):
            return input_value.strip()
        return " ".join(input_value).strip()


async def _maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value
