import asyncio
import contextvars
import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import typer
from glyff.exceptions import YieldException
from rich.console import Console
from sefia import get_context
from sefios import SefiaScope

from .human_input import ChatHumanInputAdapter
from .session import ChatSession, SessionManager
from .ui import print_session_interrupted_hint
from .workflow import WorkflowState

T = TypeVar("T")


@dataclass(frozen=True)
class SefiaCLIInvocation:
    """Runtime information for the currently running Sefia CLI command."""

    session_id: str
    is_new: bool
    input_accepted: bool = False
    current_input: str | None = None
    initial_input: str | None = None


class SefiaCLI:
    """Runs Typer command callbacks inside a Sefia session context."""

    def __init__(
        self,
        *,
        session_dir: Path,
        console: Console | None = None,
        model: str | None = None,
        stream: bool = True,
        verbose: bool = False,
        max_steps: int | None = 25,
    ):
        self.session_dir = session_dir
        self.console = console or Console()

        self.session_manager = SessionManager(session_dir)
        self.human_input = ChatHumanInputAdapter(self.console)
        self.human_input_tool = self.human_input.create_tool()

        self._sefia_scope = SefiaScope(
            session_dir=session_dir,
            model=model,
            stream=stream,
            verbose=verbose,
            max_steps=max_steps,
        )
        self._scoped_run = self._sefia_scope(self._run_scoped_command)
        self._invocation_var: contextvars.ContextVar[SefiaCLIInvocation | None] = (
            contextvars.ContextVar("sefia_cli_invocation", default=None)
        )

    @property
    def invocation(self) -> SefiaCLIInvocation:
        invocation = self._invocation_var.get()
        if invocation is None:
            raise RuntimeError("No active Sefia CLI invocation.")
        return invocation

    @property
    def current_input(self) -> str:
        current_input = self.invocation.current_input
        if current_input is None:
            raise RuntimeError("Call 'await sefia_cli.accept_input(...)' first.")
        return current_input

    @property
    def initial_input(self) -> str:
        initial_input = self.invocation.initial_input
        if initial_input is None:
            raise RuntimeError("Call 'await sefia_cli.accept_input(...)' first.")
        return initial_input

    async def accept_input(self, input_value: str | list[str]) -> str:
        """
        Accept the current CLI input and return the workflow's initial input.

        New sessions store the input as the workflow initial input. Resumed
        sessions forward the input to a pending human interaction when present.
        """
        invocation = self.invocation
        if invocation.input_accepted:
            raise RuntimeError("Input has already been accepted for this invocation.")

        input_text = self._to_input_text(input_value)
        await self.human_input.receive_input(input_text, is_new=invocation.is_new)

        if invocation.is_new:
            await self._initialize_workflow(input_text)

        state = await self._get_workflow_state()
        updated_invocation = SefiaCLIInvocation(
            session_id=invocation.session_id,
            is_new=invocation.is_new,
            input_accepted=True,
            current_input=input_text,
            initial_input=state.initial_input,
        )
        self._invocation_var.set(updated_invocation)
        return state.initial_input

    def create_session(self) -> str:
        """Create a new active CLI session and return its ID."""
        return self.session_manager.create_new_active_session()

    def switch_session(self, session_id: str) -> str:
        """Switch the active CLI session and return its ID."""
        return self.session_manager.switch_active_session(session_id)

    def get_active_session(self) -> str | None:
        """Return the active CLI session ID, if any."""
        return self.session_manager.get_active_session_id()

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
                chat_session = self.session_manager.prepare_chat_session(session_id)
                self._print_session_status(chat_session)

                run_kwargs: dict[str, Any] = {
                    "session_id": chat_session.session_id,
                    "func": inner,
                    "bound": bound,
                    "chat_session": chat_session,
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
        chat_session: ChatSession,
    ) -> None:
        invocation = SefiaCLIInvocation(
            session_id=chat_session.session_id,
            is_new=chat_session.is_new,
        )
        token = self._invocation_var.set(invocation)

        try:
            result = func(*bound.args, **bound.kwargs)
            if inspect.isawaitable(result):
                await result

            if not self.invocation.input_accepted:
                raise RuntimeError(
                    "Sefia CLI command must call 'await sefia_cli.accept_input(...)'."
                )

        except YieldException:
            print_session_interrupted_hint()
        except Exception as e:
            self.console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
            raise typer.Exit(code=1) from e
        finally:
            self._invocation_var.reset(token)

    async def _initialize_workflow(self, initial_input: str) -> None:
        session = get_context()
        state_store = session.get_state_store("workflow_state", WorkflowState)
        state = WorkflowState.from_initial_input(initial_input)
        await state_store.save(state)

    async def _get_workflow_state(self) -> WorkflowState:
        session = get_context()
        state_store = session.get_state_store("workflow_state", WorkflowState)
        return await state_store.ensure()

    def _print_session_status(self, session: ChatSession) -> None:
        if session.source == "created":
            self.console.print(
                f"[bold]> No active session. Starting new session: {session.session_id}[/bold]"
            )
        elif session.source == "active":
            self.console.print(f"[bold]> Resuming session {session.session_id}[/bold]")

    @staticmethod
    def _to_input_text(input_value: str | list[str]) -> str:
        if isinstance(input_value, str):
            return input_value.strip()
        return " ".join(input_value).strip()
