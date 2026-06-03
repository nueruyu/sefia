from glyff import identify
from sefia import infer, tool

from .models import ArticleRequest, NewsArticle
from .tools import HumanInputTool, WebSearchTool


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
