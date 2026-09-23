import glyff

from sefia import Session, ToolFunctionInspector
from sefia.llm import (
    LLMClient,
    LLMInferenceStrategy,
    PromptRenderer,
    StructuredDataConverter,
)
from sefia.llm.result_format import ResultFormatFactory
from sefia.llm.transports import DecisionTransport
from sefia.pydantic import (
    PydanticResultFormatFactory,
    PydanticStructuredDataConverter,
    PydanticToolFunctionInspector,
)


tool_function_inspector: ToolFunctionInspector = PydanticToolFunctionInspector()
result_format_factory: ResultFormatFactory = PydanticResultFormatFactory()
structured_data_converter: StructuredDataConverter = PydanticStructuredDataConverter()


def make_strategy(
    client: LLMClient,
    renderer: PromptRenderer,
    transport: DecisionTransport,
) -> LLMInferenceStrategy:
    return LLMInferenceStrategy(
        client,
        result_format_factory=result_format_factory,
        structured_data_converter=structured_data_converter,
        prompt_renderer=renderer,
        decision_transport=transport,
    )


def make_session(client: LLMClient, glyff_session: glyff.Session) -> Session:
    return Session(
        client,
        glyff_session,
        tool_function_inspector=tool_function_inspector,
        result_format_factory=result_format_factory,
        structured_data_converter=structured_data_converter,
    )
