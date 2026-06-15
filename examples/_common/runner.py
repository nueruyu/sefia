from typing import Any, Callable, Coroutine

import typer
from glyff.exceptions import YieldException
from rich.console import Console
from sefia import get_context

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
    session_id: str,
    input_text: str,
    is_new: bool,
    model: str,
    verbose: bool,
    stream: bool = True,
) -> None:
    """
    Runs the main application logic within a Sefia session context.

    This function handles session initialization, state management, human input
    forwarding, and exception handling for a given workflow coroutine.
    """
    try:
        session = get_context()
        pending = await session.session_store.get("pending_human_interaction", dict)
        if pending and not is_new:
            interaction_id = pending["id"]
            await session.session_store.set(
                f"human_input__{interaction_id}", input_text, str
            )

        if is_new:
            await initialize_workflow(input_text)

        state_store = session.get_state_store("workflow_state", WorkflowState)
        state = await state_store.ensure()

        await workflow_coro(state.initial_input)

    except YieldException:
        print_session_interrupted_hint()
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        raise typer.Exit(code=1) from e
