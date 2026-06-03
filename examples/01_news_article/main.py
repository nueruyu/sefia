from textwrap import dedent

from glyff import engrave, identify
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from sefia import infer, tool
from sefia.tools.web import WebSearchTool

from .cli import create_app
from .session import SessionState
from .tools import HumanInputTool

console = Console()


class NewsArticle(BaseModel):
    """Represents a finalized news article."""

    title: str
    summary: str
    sources: list[str]

    def to_markdown(self):
        return (
            dedent(
                """
                ## Title
                {title}

                ## Summary
                {summary}

                ## Sources
                {sources}
                """
            )
            .strip()
            .format(
                title=self.title,
                summary=self.summary,
                sources="\n".join(f"- {source}" for source in self.sources)
                or "- (none)",
            )
        )


class ArticleRequest(BaseModel):
    """Represents a clarified request for a news article."""

    topic: str
    angle: str
    audience: str
    requirements: list[str]

    def to_markdown(self) -> str:
        return (
            dedent(
                """
                Topic: {topic}
                Angle: {angle}
                Audience: {audience}
                Requirements:
                {requirements}
                """
            )
            .strip()
            .format(
                topic=self.topic,
                angle=self.angle,
                audience=self.audience,
                requirements="\n".join(
                    f"- {requirement}" for requirement in self.requirements
                )
                or "- (none)",
            )
        )


@identify("RequirementsClarifier")
class RequirementsClarifier:
    def __init__(self, human_input: HumanInputTool):
        self._human_input = human_input

    @infer()
    async def clarify_request(self, user_request: str) -> ArticleRequest:
        """
        Clarify the user's initial request before any research or writing begins.

        Your goal is to produce a concrete article brief for the downstream
        researcher and writer.

        If the request lacks important details, ask the user one focused question
        at a time using the HumanInputTool. Repeat this until all critical
        ambiguities are resolved.

        Critical details include:
        1. The article topic or subject.
        2. The intended angle or emphasis.
        3. The target audience.
        4. Any must-include points, constraints, or exclusions.

        Do not ask about optional details if the user's request is already clear
        enough to proceed. Use reasonable defaults when they do not materially
        change the result.
        """
        ...


@identify("Researcher")
class Researcher:
    def __init__(self, web_search: WebSearchTool):
        self._web = web_search

    @tool
    @infer()
    async def research_topic(self, article_request: ArticleRequest) -> list[str]:
        """
        Research the clarified article request to find relevant online sources.
        Your goal is to return a list of high-quality URLs related to the request.

        **CRITICAL INSTRUCTIONS:**
        1. You MUST use the `WebSearchTool` tool to find the URLs.
        2. Do NOT answer from your own knowledge.
        3. The final answer MUST be a list of strings, where each string is a valid URL.
        """
        ...


@identify("NewsWriter")
class NewsWriter:
    def __init__(self, human_input: HumanInputTool, researcher: Researcher):
        self._human_input = human_input
        self._researcher = researcher

    @infer()
    async def write_article(
        self, article_request: ArticleRequest, sources: list[str]
    ) -> NewsArticle:
        """
        Write a news article for the clarified request, using the provided sources.
        1. Briefly review the sources to understand the key points.
        2. Write a draft that follows the topic, angle, audience, and requirements.
        3. Ask the user for feedback on the draft's direction using the HumanInputTool.
        4. If the user's feedback reveals missing information or requires additional
           evidence, ask the Researcher to gather more sources before finalizing.
        5. Finalize the article based on the user's feedback, incorporating their
           suggestions and any additional research.
        6. Return the final article as a NewsArticle object.
        """
        ...


clarifier = RequirementsClarifier(HumanInputTool())
researcher = Researcher(WebSearchTool())
writer = NewsWriter(HumanInputTool(), researcher)


@engrave
async def _clarify(initial_topic: str) -> ArticleRequest:
    console.print("[bold]> Stage 1: Clarifying request...[/bold]")
    article_request = await clarifier.clarify_request(initial_topic)

    console.print("[dim]   -> Clarified request:[/dim]")
    console.print(Markdown(article_request.to_markdown()))
    return article_request


@engrave
async def _research(article_request: ArticleRequest) -> list[str]:
    console.print("[bold]> Stage 2: Researching topic...[/bold]")
    sources = await researcher.research_topic(article_request)
    console.print(f"[dim]   -> Found sources: {sources}[/dim]")
    return sources


@engrave
async def _write(article_request: ArticleRequest, sources: list[str]) -> NewsArticle:
    console.print("[bold]> Stage 3: Writing article...[/bold]")
    return await writer.write_article(article_request=article_request, sources=sources)


async def workflow(state: SessionState) -> None:
    article_request = await _clarify(state.initial_topic)
    sources = await _research(article_request)
    article = await _write(article_request, sources)

    console.print(
        Panel(
            Markdown(article.to_markdown()),
            title="FINAL ARTICLE",
            border_style="green",
            expand=False,
        )
    )


if __name__ == "__main__":
    create_app(workflow)()
