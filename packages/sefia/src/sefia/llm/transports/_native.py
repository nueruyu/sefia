from typing import cast

from typing_extensions import final, override

from ...inference import HistoryItem, ToolCallsDecision
from .._client import LLMClient, LLMResponseDecodingError
from .._messages import Message, ToolCall
from .._prompt_renderer import PromptRenderer
from ..json_schema import JsonSchemaDocument
from ..llm_output import LLMOutput, LLMOutputData
from ..step_decision import DecisionSpec, StepDecisionMode, StepTool, ToolSchemaSource
from ._base import (
    DecisionDecodingError,
    DecisionObserver,
    DecisionRequest,
    DecisionResponse,
    DecisionTransport,
)

_RESULT_TOOL_NAME = "return_result"


@final
class NativeDecisionTransport(DecisionTransport):
    """Represents decisions with native tool calls and tool-result messages."""

    @override
    async def request_decision(
        self,
        client: LLMClient,
        prompt_renderer: PromptRenderer,
        request: DecisionRequest,
        observer: DecisionObserver,
        stream: bool,
    ) -> DecisionResponse:
        tools, result_tool = _native_tools(request.spec)
        prompt = prompt_renderer.render(
            request.to_prompt(
                _native_response_instructions(request.spec, result_tool),
                tools=(),
                history=(),
            )
        )
        await observer.before_request(prompt)

        try:
            response = await client.complete(
                messages=[
                    Message(role="user", content=prompt),
                    *_native_history(request.history, prompt_renderer),
                ],
                tools=tools,
                decision_model=None,
                stream_callback=observer.response_text if stream else None,
                output_callback=observer.output if stream else None,
                reasoning_callback=observer.reasoning_text if stream else None,
            )
        except LLMResponseDecodingError as error:
            raise DecisionDecodingError(error.response, str(error)) from error
        try:
            output = _decode_native_decision(response.tool_calls, result_tool)
        except ValueError as error:
            raise DecisionDecodingError(response, str(error)) from error
        return DecisionResponse(output=output, raw=response)


def _native_response_instructions(
    decision: DecisionSpec,
    result_tool: StepTool | None,
) -> str:
    if decision.mode is StepDecisionMode.TOOLS_REQUIRED:
        action = "Call one or more available tools."
    elif decision.mode is StepDecisionMode.RESULT_ONLY:
        assert result_tool is not None
        action = f"Call `{result_tool.name}` with the final result."
    else:
        assert result_tool is not None
        action = (
            "Call available tools when needed. When the task is complete, "
            f"call `{result_tool.name}` with the final result."
        )
    return "\n".join(
        [
            action,
            "Do not describe a tool call or answer with text.",
        ]
    )


def _native_tools(decision: DecisionSpec) -> tuple[list[StepTool], StepTool | None]:
    tools = list(decision.tools)
    if decision.result is None:
        return tools, None

    result_tool = StepTool(
        name=_available_result_name({tool.name for tool in tools}),
        description="Return the final result when the task is complete.",
        arguments=JsonSchemaDocument.from_mapping(
            {
                "type": "object",
                "properties": {"result": decision.result.schema.to_dict()},
                "required": ["result"],
                "additionalProperties": False,
            }
        ),
        schema_source=ToolSchemaSource.GENERATED,
    )
    return [*tools, result_tool], result_tool


def _available_result_name(existing: set[str]) -> str:
    name = _RESULT_TOOL_NAME
    suffix = 2
    while name in existing:
        name = f"{_RESULT_TOOL_NAME}_{suffix}"
        suffix += 1
    return name


def _native_history(
    history: tuple[HistoryItem, ...],
    prompt_renderer: PromptRenderer,
) -> list[Message]:
    messages: list[Message] = []
    for item in history:
        if isinstance(item, ToolCallsDecision):
            messages.append(
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id=call.id,
                            name=call.name,
                            arguments=LLMOutput.from_data(
                                cast(LLMOutputData, call.arguments)
                            ),
                        )
                        for call in item.calls
                    ],
                )
            )
        else:
            messages.append(
                Message(
                    role="tool",
                    content=prompt_renderer.render_tool_result(item.result),
                    tool_call_id=item.tool_call_id,
                )
            )
    return messages


def _decode_native_decision(
    calls: list[ToolCall],
    result_tool: StepTool | None,
) -> LLMOutput:
    if not calls:
        raise ValueError("LLM did not call a native decision tool.")

    result_name = result_tool.name if result_tool is not None else None
    result_calls = [call for call in calls if call.name == result_name]
    if result_calls:
        if len(calls) != 1:
            raise ValueError("The result tool cannot be combined with other calls.")
        arguments = result_calls[0].arguments.to_object("result tool arguments")
        if set(arguments) != {"result"}:
            raise ValueError("The result tool requires exactly the 'result' field.")
        return LLMOutput.from_object(
            {
                "decision": LLMOutput.from_json("result"),
                "result": arguments["result"],
            }
        )

    return LLMOutput.from_object(
        {
            "decision": LLMOutput.from_json("tool_calls"),
            "tool_calls": LLMOutput.from_array(
                LLMOutput.from_object(
                    {
                        "name": LLMOutput.from_json(call.name),
                        "arguments": LLMOutput.from_object(
                            call.arguments.to_object("tool arguments")
                        ),
                    }
                )
                for call in calls
            ),
        }
    )
