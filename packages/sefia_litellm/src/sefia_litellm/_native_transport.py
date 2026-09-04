from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from typing_extensions import final, override

from sefia.llm import (
    LLMClient,
    LLMOutput,
    Message,
    PromptRenderer,
    ToolCall,
)
from sefia.llm.step_decision import DecisionSpec, StepDecisionMode
from sefia.llm.transports import (
    DecisionDecodingError,
    DecisionObserver,
    DecisionRequest,
    DecisionResponse,
    DecisionTransport,
)

from ._schema._value_format import StructuredValueFormat

_RESULT_TOOL_NAME = "return_result"


@dataclass(frozen=True)
class _NativeTool:
    name: str
    value: StructuredValueFormat
    definition: dict[str, Any]


@final
class NativeDecisionTransport(DecisionTransport):
    """Uses provider-native function calls for every decision outcome."""

    @override
    async def request_decision(
        self,
        client: LLMClient,
        prompt_renderer: PromptRenderer,
        request: DecisionRequest,
        observer: DecisionObserver,
        stream: bool,
    ) -> DecisionResponse:
        tools, result_tool = _tools(request.spec)
        definitions = [tool.definition for tool in tools.values()]
        if result_tool is not None:
            definitions.append(result_tool.definition)
        prompt = prompt_renderer.render(
            request.to_prompt(_native_response_instructions(request.spec, result_tool))
        )
        await observer.before_request(prompt)

        response = await client.complete(
            messages=[Message(role="user", content=prompt)],
            tools=definitions,
            decision_model=None,
            stream_callback=observer.response_text if stream else None,
            output_callback=observer.output if stream else None,
            reasoning_callback=observer.reasoning_text if stream else None,
        )
        try:
            output = _decode(response.tool_calls, tools, result_tool)
        except ValueError as error:
            raise DecisionDecodingError(response, str(error)) from error
        return DecisionResponse(output=output, raw=response)


def _native_response_instructions(
    decision: DecisionSpec,
    result_tool: _NativeTool | None,
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


def _tools(
    decision: DecisionSpec,
) -> tuple[dict[str, _NativeTool], _NativeTool | None]:
    tools: dict[str, _NativeTool] = {}
    for tool in decision.tools:
        value = StructuredValueFormat.from_tool(tool)
        tools[tool.name] = _NativeTool(
            name=tool.name,
            value=value,
            definition=_definition(tool.name, tool.description, value.schema),
        )
    if decision.result is None:
        return tools, None

    name = _available_result_name(set(tools))
    result = StructuredValueFormat.from_generated_schema(decision.result.schema)
    schema = {
        "type": "object",
        "properties": {"result": result.schema},
        "required": ["result"],
        "additionalProperties": False,
    }
    return tools, _NativeTool(
        name=name,
        value=result,
        definition=_definition(
            name, "Return the final result when the task is complete.", schema
        ),
    )


def _definition(name: str, description: str, parameters: object) -> dict[str, Any]:
    function: dict[str, Any] = {"name": name, "parameters": parameters}
    if description:
        function["description"] = description
    return {"type": "function", "function": function}


def _available_result_name(existing: set[str]) -> str:
    name = _RESULT_TOOL_NAME
    suffix = 2
    while name in existing:
        name = f"{_RESULT_TOOL_NAME}_{suffix}"
        suffix += 1
    return name


def _decode(
    calls: list[ToolCall],
    tools: dict[str, _NativeTool],
    result_tool: _NativeTool | None,
) -> LLMOutput:
    if not calls:
        raise ValueError("LLM did not call a native decision tool.")

    result = result_tool
    result_calls = [
        call for call in calls if _call_name(call) == getattr(result, "name", None)
    ]
    if result_calls:
        if len(calls) != 1:
            raise ValueError("The result tool cannot be combined with other calls.")
        arguments = _decode_arguments(result_calls[0])
        if set(arguments) != {"result"}:
            raise ValueError("The result tool requires exactly the 'result' field.")
        assert result is not None
        value = result.value.decode(arguments["result"])
        return LLMOutput.from_object(
            {"decision": LLMOutput.from_json("result"), "result": value}
        )

    output_calls: list[LLMOutput] = []
    for call in calls:
        name = _call_name(call)
        tool = tools.get(name)
        arguments = LLMOutput.from_object(_decode_arguments(call))
        if tool is not None:
            arguments = tool.value.decode(arguments)
        output_calls.append(
            LLMOutput.from_object(
                {"name": LLMOutput.from_json(name), "arguments": arguments}
            )
        )
    return LLMOutput.from_object(
        {
            "decision": LLMOutput.from_json("tool_calls"),
            "tool_calls": LLMOutput.from_array(output_calls),
        }
    )


def _call_name(call: ToolCall) -> str:
    name = call.function.get("name")
    if not isinstance(name, str):
        raise ValueError("Native tool call has no function name.")
    return name


def _arguments_json(call: ToolCall) -> str:
    raw = call.function.get("arguments")
    if not isinstance(raw, str):
        raise ValueError(
            f"Native tool call {_call_name(call)!r} has no JSON arguments."
        )
    return raw


def _decode_arguments(call: ToolCall) -> dict[str, LLMOutput]:
    return LLMOutput.parse_json(_arguments_json(call)).to_object(
        f"Native tool call {_call_name(call)!r} arguments"
    )


__all__ = ["NativeDecisionTransport"]
