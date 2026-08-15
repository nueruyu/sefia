from importlib import import_module

from sefia._tool_system import SignatureToolEntry
from sefia.llm._execution_directors import OutputOnlyDirector, ToolEnabledDirector
from sefia.pydantic import PydanticModelBackend
from sefios.tools import WebSearch

news_agents = import_module("examples.01_news_article.agents")
news_models = import_module("examples.01_news_article.models")
quality_models = import_module("examples.02_code_quality.models")


def test_news_writer_schema_composes_nested_research_tool_types() -> None:
    backend = PydanticModelBackend()
    researcher = news_agents.Researcher(WebSearch())
    research = researcher.research_topic
    tool = SignatureToolEntry(
        research,
        name=backend.tool_name(research),
        schema_source=research,
        inspector=backend,
    )

    schema = ToolEnabledDirector(
        backend, news_models.NewsArticle, [tool]
    ).build_decision_schema()

    assert schema["additionalProperties"] is False


def test_code_quality_report_schema_lowers_perspective_mapping() -> None:
    schema = OutputOnlyDirector(
        PydanticModelBackend(), quality_models.QualityReport, []
    ).build_decision_schema()

    report = schema["$defs"]["QualityReport"]
    perspective_issues = report["properties"]["issues_by_perspective"]
    assert perspective_issues["type"] == "array"
    assert perspective_issues["items"]["additionalProperties"] is False
