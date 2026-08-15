import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from sefios import SQLitePersistence, get_state
from sefios.cli import SefiaCLI
from sefios.sessions import FileActiveSessionStore
from typer.models import ArgumentInfo, OptionInfo

from .._common.policies import VerbosePolicy
from .._common.typer_utils import add_session_commands, async_command
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
from .authoring import engrave
from .models import (
    CodeIssue,
    ProjectScope,
    ProjectUnderstanding,
    QualityReport,
    ReviewPerspective,
)
from .rendering import render_quality_report
from .tools import Files, Git

console = Console()
SESSION_DIR = Path(__file__).parent / ".local"
sefia_cli = SefiaCLI(
    stream=True,
    persistence=SQLitePersistence(SESSION_DIR / "sessions.sqlite3"),
    active_session_store=FileActiveSessionStore(SESSION_DIR / "active_session.txt"),
)
input_tool = sefia_cli.input_tool

git_tool = Git()
file_tool = Files()

scoping_agent = ScopingAgent(input_tool)
understanding_agent = UnderstandingAgent(file_tool)
review_scoping_agent = ReviewScopingAgent(input_tool)
reporting_agent = ReportingAgent()

review_agents = {
    ReviewPerspective.CODING_STYLE: CodingStyleAuditor(),
    ReviewPerspective.DESIGN_PRINCIPLES: DesignPrincipleArchitect(),
    ReviewPerspective.MAINTAINABILITY: MaintainabilityAssessor(),
    ReviewPerspective.DEPENDENCIES: DependencySpecialist(),
}

app = typer.Typer(help="A multi-agent workflow for code quality review.")


@engrave
async def _define_scope() -> ProjectScope:
    console.print("[bold]> Stage 1: Defining scope...[/bold]")
    return await scoping_agent.define_scope()


@engrave
async def _understand_project(scope: ProjectScope) -> ProjectUnderstanding:
    console.print("\n[bold]> Stage 2: Understanding project...[/bold]")
    file_paths = await git_tool.list_tracked_files(scope.project_path)
    understanding_store = get_state().get(ProjectUnderstanding)
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
@async_command
async def chat(
    message: Annotated[
        list[str],
        ArgumentInfo(
            default=...,
            help="The input for a new session, or an answer to resume an existing one.",
        ),
    ],
    reply_to: Annotated[
        str | None,
        OptionInfo(
            default=...,
            param_decls=("--reply-to",),
            help="The input interaction ID to answer.",
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        OptionInfo(
            default=...,
            param_decls=("--session-id",),
            help="The session ID to use. If not provided, uses the active session.",
        ),
    ] = None,
    model: Annotated[
        str,
        OptionInfo(
            default=...,
            param_decls=("--model",),
            help="The LLM model to use. Can also be set via EXAMPLE_DEFAULT_MODEL env var.",
            envvar="EXAMPLE_DEFAULT_MODEL",
        ),
    ] = "gpt-4o",
    verbose: Annotated[
        bool,
        OptionInfo(
            default=...,
            param_decls=("--verbose",),
            help="Enable verbose output for debugging, including LLM prompts.",
        ),
    ] = False,
) -> None:
    """Start a new workflow or provide an answer to continue the current session."""
    async with sefia_cli.session(
        session_id=session_id,
        model=model,
        policies=[VerbosePolicy()] if verbose else None,
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


add_session_commands(app, sefia_cli)

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    app()
