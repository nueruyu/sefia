from typing import Any, Callable, Coroutine

import typer
from glyff.exceptions import YieldException
from rich.console import Console
from sefia import get_context

from .human_input import ChatHumanInputAdapter
from .ui import print_session_interrupted_hint
from .workflow import WorkflowState

console = Console()


async def initialize_workflow(initial_input: str) -> None:
    """Initializes the workflow state for a new session."""
    session = get_context()
    state_store = session.get_state_store("workflow_state", WorkflowState)
    state = WorkflowState.from_initial_input(initial_input)
    await state_store.save(state)


async def run_workflow(
    *,
    workflow_coro: Callable[[str], Coroutine[Any, Any, None]],
    input_text: str,
    is_new: bool,
    human_input: ChatHumanInputAdapter | None = None,
) -> None:
    """
    Runs the main application logic within a Sefia session context.

    This function handles session initialization, state management, human input
    forwarding, and exception handling for a given workflow coroutine.
    """
    try:
        if human_input is not None:
            await human_input.receive_input(input_text, is_new=is_new)

        if is_new:
            await initialize_workflow(input_text)

        session = get_context()
        state_store = session.get_state_store("workflow_state", WorkflowState)
        state = await state_store.ensure()

        await workflow_coro(state.initial_input)

    except YieldException:
        print_session_interrupted_hint()
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        raise typer.Exit(code=1) from e
