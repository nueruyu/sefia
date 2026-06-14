import asyncio
import sys
from pathlib import Path
from textwrap import dedent

import typer
from glyff import engrave
from glyff.exceptions import YieldException
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from sefia import get_context
from sefios import SefiaScope
from sefios.tools import HumanInputTool
from typing_extensions import Annotated

from .._common.session import SessionManager, create_session_cli
from .._common.workflow import WorkflowState
from .agents import (
    CodingStyleAuditor,
    DependencySpecialist,
    DesignPrincipleArchitect,
    MaintainabilityAssessor,
    ReportingAgent,
    ReviewScopingAgent,
    ScopingAgent,
    UnderstandingAgent,
)
from .models import (
    CodeIssue,
    ProjectScope,
    ProjectUnderstanding,
    QualityReport,
    ReviewPerspective,
)
from .rendering import render_quality_report
from .tools import FileTool, GitTool

human_input_tool = HumanInputTool()
git_tool = GitTool()
file_tool = FileTool()

scoping_agent = ScopingAgent(human_input_tool)
understanding_agent = UnderstandingAgent(file_tool)
review_scoping_agent = ReviewScopingAgent(human_input_tool)
reporting_agent = ReportingAgent()

review_agents = {
    ReviewPerspective.CODING_STYLE: CodingStyleAuditor(),
    ReviewPerspective.DESIGN_PRINCIPLES: DesignPrincipleArchitect(),
    ReviewPerspective.MAINTAINABILITY: MaintainabilityAssessor(),
    ReviewPerspective.DEPENDENCIES: DependencySpecialist(),
}

SESSION_DIR = Path(__file__).parent / ".local"
console = Console()

sefia_scope = SefiaScope(session_dir=SESSION_DIR)
app = typer.Typer(help="A multi-agent workflow for code quality review.")
session_manager = SessionManager(SESSION_DIR)
create_session_cli(app, session_manager)


def _print_session_interrupted_hint() -> None:
    hint = dedent(
        """
        Session interrupted to wait for your input.
        To resume, run the script again with your answer.
        """
    ).strip()
    console.print()
    console.print(Panel(hint, title="WAITING FOR INPUT", border_style="yellow"))


async def _initialize_workflow(initial_input: str) -> None:
    """Initializes the workflow state for a new session."""
    session = get_context()
    state_store = session.get_state_store("workflow_state", WorkflowState)
    state = WorkflowState.from_initial_input(initial_input)
    await state_store.save(state)


@engrave
async def _define_scope(user_request: str) -> ProjectScope:
    console.print("[bold]> Stage 1: Defining scope...[/bold]")
    return await scoping_agent.define_scope(user_request)


async def _understand_project(scope: ProjectScope) -> ProjectUnderstanding:
    console.print("\n[bold]> Stage 2: Understanding project...[/bold]")
    file_paths = await git_tool.list_tracked_files(scope.project_path)
    understanding_store = get_context().get_state_store(
        "project_understanding", ProjectUnderstanding
    )
    understanding = await understanding_store.ensure()

    for iteration in range(5):
        console.print(f"\n> Understanding Iteration {iteration + 1}...")
        previous_understanding = understanding.copy()
        understanding = await understanding_agent.deepen_understanding(
            understanding,
            file_paths,
            scope.project_path,
        )
        await understanding_store.save(understanding)

        if understanding == previous_understanding:
            console.print("[green]> Understanding sufficient.[/green]")
            break

    return understanding


@engrave
async def _confirm_review_files(
    scope: ProjectScope,
    understanding: ProjectUnderstanding,
) -> list[str]:
    console.print("\n[bold]> Stage 3: Confirming files for review...[/bold]")
    all_files = await git_tool.list_tracked_files(scope.project_path)
    review_files = await review_scoping_agent.propose_and_confirm_review_files(
        understanding,
        scope,
        all_files,
    )
    tracked_files = set(all_files)
    return list(dict.fromkeys(path for path in review_files if path in tracked_files))


async def _run_reviews(
    review_files: list[str],
    project_path: str,
) -> list[CodeIssue]:
    console.print("\n[bold]> Stage 4: Running reviews...[/bold]")
    full_to_relative = {str(Path(project_path) / path): path for path in review_files}
    contents = await file_tool.read_files(list(full_to_relative))
    all_issues: list[CodeIssue] = []

    async def _review_perspective(
        perspective: ReviewPerspective,
    ) -> tuple[ReviewPerspective, list[CodeIssue]]:
        console.print(f"> Reviewing from perspective: {perspective.value}...")
        agent = review_agents[perspective]
        relative_contents = {
            full_to_relative[path]: content for path, content in contents.items()
        }
        return perspective, await agent.review(relative_contents)

    review_results = await asyncio.gather(
        *(_review_perspective(perspective) for perspective in review_agents)
    )

    for perspective, issues in review_results:
        for issue in issues:
            issue.perspective = perspective.value
        all_issues.extend(issues)

    return all_issues


@engrave
async def _create_report(
    issues: list[CodeIssue],
    understanding: ProjectUnderstanding,
) -> QualityReport:
    console.print("\n[bold]> Stage 5: Creating report...[/bold]")
    return await reporting_agent.create_report(issues, understanding)


@sefia_scope
async def _run_chat(
    *,
    session_id: str,
    input_text: str,
    is_new: bool,
    model: str,
    verbose: bool,
    stream: bool = True,
) -> None:
    try:
        session = get_context()
        pending = await session.session_store.get("pending_human_interaction", dict)
        if pending and not is_new:
            interaction_id = pending["id"]
            await session.session_store.set(
                f"human_input__{interaction_id}", input_text, str
            )

        if is_new:
            await _initialize_workflow(input_text)

        state_store = session.get_state_store("workflow_state", WorkflowState)
        state = await state_store.ensure()

        scope = await _define_scope(state.initial_input)
        understanding = await _understand_project(scope)
        review_files = await _confirm_review_files(scope, understanding)

        if not review_files:
            console.print("[yellow]No files selected for review. Exiting.[/yellow]")
            return

        issues = await _run_reviews(review_files, scope.project_path)
        report = await _create_report(issues, understanding)

        console.print(
            Panel(
                Markdown(render_quality_report(report)),
                title="FINAL CODE QUALITY REPORT",
                border_style="green",
            )
        )
    except YieldException:
        _print_session_interrupted_hint()
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def chat(
    message: Annotated[
        list[str] | None,
        typer.Argument(
            help="The input for a new session, or an answer to resume an existing one."
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            help="The session ID to use. If not provided, uses the active session."
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            help="The LLM model to use. Can also be set via EXAMPLE_DEFAULT_MODEL env var.",
            envvar="EXAMPLE_DEFAULT_MODEL",
        ),
    ] = "gpt-4o",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Enable verbose output for debugging, including LLM prompts.",
        ),
    ] = False,
):
    """
    Start a new workflow or provide an answer to continue the current session.
    """
    input_text = " ".join(message or []).strip()

    if not input_text and not sys.stdin.isatty():
        input_text = sys.stdin.read().strip()

    if not input_text:
        console.print("[bold red]Error:[/bold red] Message cannot be empty.")
        raise typer.Exit(code=1)

    is_new = session_id is None
    if is_new:
        session_id = session_manager.get_active_session_id()
        if session_id is None:
            session_id = session_manager.create_new_session_id()
            console.print(
                f"[bold]> No active session. Starting new session: {session_id}[/bold]"
            )
            session_manager.set_active_session_id(session_id)
        else:
            console.print(f"[bold]> Resuming session {session_id}[/bold]")
            is_new = False
    else:
        is_new = False

    asyncio.run(
        _run_chat(
            session_id=session_id,
            input_text=input_text,
            is_new=is_new,
            model=model,
            verbose=verbose,
            stream=True,
        )
    )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    app()
