import asyncio
from pathlib import Path

import typer
from glyff import engrave
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from sefia import get_context
from typing_extensions import Annotated

from .._common.sefia_cli import SefiaCLI
from .._common.session import UnknownSessionError
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

console = Console()
SESSION_DIR = Path(__file__).parent / ".local"
sefia_cli = SefiaCLI(session_dir=SESSION_DIR, stream=True)
human_input_tool = sefia_cli.human_input_tool

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

app = typer.Typer(help="A multi-agent workflow for code quality review.")
session_app = typer.Typer(help="Manage sessions.")
app.add_typer(session_app, name="session")


@session_app.command("new")
def new_session():
    """Create a new session and make it active."""
    session_id = sefia_cli.create_session()
    console.print(f"[bold]> Created and switched to new session: {session_id}[/bold]")


@session_app.command("switch")
def switch_session(
    session_id: Annotated[
        str,
        typer.Argument(help="The ID of the session to switch to."),
    ],
):
    """Switch the active session."""
    try:
        session_id = sefia_cli.switch_session(session_id)
    except UnknownSessionError as e:
        console.print(f"[bold red]> Unknown session:[/bold red] {e.session_id}")
        raise typer.Exit(code=1) from e

    console.print(f"[bold]> Switched active session to: {session_id}[/bold]")


@engrave
async def _define_scope() -> ProjectScope:
    console.print("[bold]> Stage 1: Defining scope...[/bold]")
    return await scoping_agent.define_scope()


@engrave
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


@engrave
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


@app.command()
def chat(
    message: Annotated[
        list[str],
        typer.Argument(
            help="The input for a new session, or an answer to resume an existing one."
        ),
    ],
    reply_to: Annotated[
        str | None,
        typer.Option(
            "--reply-to",
            help="The human input interaction ID to answer.",
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="The session ID to use. If not provided, uses the active session.",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
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
    """Start a new workflow or provide an answer to continue the current session."""
    asyncio.run(
        _chat_async(
            message=message,
            reply_to=reply_to,
            session_id=session_id,
            model=model,
            verbose=verbose,
        )
    )


async def _chat_async(
    *,
    message: list[str],
    reply_to: str | None,
    session_id: str | None,
    model: str,
    verbose: bool,
) -> None:
    async with sefia_cli.session(
        session_id=session_id,
        model=model,
        verbose=verbose,
    ) as session:
        await session.accept_input(message, reply_to=reply_to)

        scope = await _define_scope()
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


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    app()
