import glyff
from sefia import Session, ToolFunctionInspector
from sefia.llm import (
    JsonMaterializer,
    LLMClient,
    LLMInferenceStrategy,
    PromptRenderer,
)
from sefia.llm.result_format import ResultFormatFactory
from sefia.llm.transports import DecisionTransport
from sefia.pydantic import (
    PydanticJsonMaterializer,
    PydanticResultFormatFactory,
    PydanticToolFunctionInspector,
)
from sefia.tool_collectors import DefaultToolCollector

tool_function_inspector: ToolFunctionInspector = PydanticToolFunctionInspector()
result_format_factory: ResultFormatFactory = PydanticResultFormatFactory()
json_materializer: JsonMaterializer = PydanticJsonMaterializer()


def make_strategy(
    client: LLMClient,
    renderer: PromptRenderer,
    transport: DecisionTransport,
) -> LLMInferenceStrategy:
    return LLMInferenceStrategy(
        client,
        result_format_factory=result_format_factory,
        json_materializer=json_materializer,
        prompt_renderer=renderer,
        decision_transport=transport,
    )


def make_session(client: LLMClient, glyff_session: glyff.Session) -> Session:
    return Session(
        client,
        glyff_session,
        tool_collector=DefaultToolCollector(inspector=tool_function_inspector),
        result_format_factory=result_format_factory,
        json_materializer=json_materializer,
    )
