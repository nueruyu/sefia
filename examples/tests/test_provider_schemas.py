from importlib import import_module

from sefia._tool_system import SignatureToolEntry, ToolEntry
from sefia.llm.json_schema import SchemaNode
from sefia.llm.step_decision import StepDecisionModel, StepDecisionSpec
from sefia.pydantic import PydanticModelBackend
from sefia_litellm._schema import DecisionEnvelopeFormat
from sefios.tools import WebSearch

news_agents = import_module("examples.01_news_article.agents")
news_models = import_module("examples.01_news_article.models")
quality_models = import_module("examples.02_code_quality.models")


def _decision_schema(output_type: object, tools: list[ToolEntry]):
    spec = StepDecisionSpec.for_inference(
        name="StepDecision", output_type=output_type, tools=tools
    )
    return StepDecisionModel.from_spec(spec, PydanticModelBackend())


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

    model = _decision_schema(news_models.NewsArticle, [tool])
    schema = DecisionEnvelopeFormat.from_model(model).schema.to_dict()

    assert schema["additionalProperties"] is False


def test_code_quality_report_schema_lowers_perspective_mapping() -> None:
    model = _decision_schema(quality_models.QualityReport, [])
    schema = DecisionEnvelopeFormat.from_model(model).schema.to_dict()

    payload = SchemaNode(schema).properties()["payload"]
    perspective_issues = payload.properties()["result"].properties()[
        "issues_by_perspective"
    ]
    assert perspective_issues.type == "array"
    items = perspective_issues.child("items")
    assert items is not None
    assert items.additional_properties() is False
