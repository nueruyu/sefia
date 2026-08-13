from sefios import domain, Tools
from sefios.tools import Input, WebSearch

from .models import ArticleRequest, NewsArticle


class RequirementsClarifier:
    _input: Tools[Input]

    def __init__(self, input_tool: Input):
        self._input = input_tool

    @domain("examples.01_news_article.agents", version="1").infer(
        name="RequirementsClarifier.clarify_request"
    )
    async def clarify_request(self) -> ArticleRequest:
        """
        Clarify the user's request for a news article before research or writing.

        Your goal is to produce a concrete article brief for the downstream
        researcher and writer.

        First, use the Input tool to obtain the user's initial article
        request. Treat that answer as the source request; do not ask the user to
        restate it. Then, if the request lacks important details, ask one
        focused follow-up question at a time. Repeat this only until critical
        ambiguities are resolved.

        Critical details include:
        1. The article topic or subject.
        2. The intended angle or emphasis.
        3. The target audience.
        4. Any must-include points, constraints, or exclusions.

        Do not ask about optional details if the user's request is already clear
        enough to proceed. Use reasonable defaults when they do not materially
        change the result, especially for angle, audience, language, and
        exclusions.
        """
        ...


class Researcher:
    _web: Tools[WebSearch]

    def __init__(self, web_search: WebSearch):
        self._web = web_search

    @domain("examples.01_news_article.agents", version="1").infer(
        name="Researcher.research_topic"
    )
    async def research_topic(self, article_request: ArticleRequest) -> list[str]:
        """
        Research the clarified article request to find relevant online sources.
        Your goal is to return a list of high-quality URLs related to the request.

        **CRITICAL INSTRUCTIONS:**
        1. You MUST use the `WebSearch` tool to find the URLs.
        2. Do NOT answer from your own knowledge.
        3. The final answer MUST be a list of strings, where each string is a valid URL.
        """
        ...


class NewsWriter:
    _input: Tools[Input]
    _researcher: Tools[Researcher]

    def __init__(self, input_tool: Input, researcher: Researcher):
        self._input = input_tool
        self._researcher = researcher

    @domain("examples.01_news_article.agents", version="1").infer(
        name="NewsWriter.write_article"
    )
    async def write_article(
        self, article_request: ArticleRequest, sources: list[str]
    ) -> NewsArticle:
        """
        Write a news article for the clarified request, using the provided sources.
        1. Briefly review the sources to understand the key points.
        2. Write a draft that follows the topic, angle, audience, and requirements.
        3. Ask the user for feedback on the draft's direction using the Input tool
           at most once.
        4. After receiving any feedback, apply it and return the final NewsArticle.
           Do not ask another Input tool question unless the feedback is
           impossible to apply without a specific missing fact from the user.
        5. If the feedback asks to see the draft, change language, continue, add
           a point, remove a point, or proceed, treat that as actionable feedback
           and finalize the article instead of asking for confirmation again.
        6. If the user's feedback reveals missing information or requires additional
           evidence, ask the Researcher to gather more sources before finalizing,
           then return the final NewsArticle without another user confirmation.
        """
        ...
