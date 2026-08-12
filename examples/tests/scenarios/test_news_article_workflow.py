from importlib import import_module
from unittest.mock import AsyncMock

import pytest
from sefios.cli import SefiaCLI

# main.py uses ``from .._common ...`` imports, so it must be loaded with the
# full ``examples.`` package prefix (two parent levels), unlike the rendering
# and tools modules which are loaded relative to the examples directory.
main = import_module("examples.01_news_article.main")
models = import_module("examples.01_news_article.models")


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    """Point the example's module-level CLI at a throwaway session directory."""
    cli = SefiaCLI(session_dir=tmp_path / "sessions", model="gpt-4o", stream=False)
    monkeypatch.setattr(main, "sefia_cli", cli)
    return main


class TestNewsArticleWorkflow:
    async def test_runs_every_stage_and_renders_article(
        self, workflow, monkeypatch, capsys
    ):
        request = models.ArticleRequest(
            topic="Generative AI",
            angle="Impact on developers",
            audience="Engineering managers",
            requirements=["Cite sources"],
        )
        article = models.NewsArticle(
            title="Generative AI Reshapes Development",
            summary="A concise summary.",
            sources=["https://example.com/a"],
        )
        clarify = AsyncMock(return_value=request)
        research = AsyncMock(return_value=["https://example.com/a"])
        write = AsyncMock(return_value=article)
        monkeypatch.setattr(workflow.clarifier, "clarify_request", clarify)
        monkeypatch.setattr(workflow.researcher, "research_topic", research)
        monkeypatch.setattr(workflow.writer, "write_article", write)

        await workflow.chat.__wrapped__(
            message=["Write about generative AI"],
            reply_to=None,
            session_id=None,
            model="gpt-4o",
            verbose=False,
        )

        clarify.assert_awaited_once()
        research.assert_awaited_once_with(request)
        write.assert_awaited_once_with(
            article_request=request, sources=["https://example.com/a"]
        )

        output = capsys.readouterr().out
        assert "Generative AI Reshapes Development" in output
        assert "A concise summary." in output

    async def test_research_output_feeds_the_writer(
        self, workflow, monkeypatch, capsys
    ):
        request = models.ArticleRequest(
            topic="t", angle="a", audience="aud", requirements=[]
        )
        sources = ["https://example.com/1", "https://example.com/2"]
        article = models.NewsArticle(title="T", summary="S", sources=sources)
        monkeypatch.setattr(
            workflow.clarifier, "clarify_request", AsyncMock(return_value=request)
        )
        monkeypatch.setattr(
            workflow.researcher, "research_topic", AsyncMock(return_value=sources)
        )
        write = AsyncMock(return_value=article)
        monkeypatch.setattr(workflow.writer, "write_article", write)

        await workflow.chat.__wrapped__(
            message=["topic"],
            reply_to=None,
            session_id=None,
            model="gpt-4o",
            verbose=False,
        )

        write.assert_awaited_once_with(article_request=request, sources=sources)
