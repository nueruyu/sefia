from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from sefia.llm import LLMOutput, LLMResponse, ToolCall
from sefia.llm.exceptions import LLMResponseDecodingError

from ._schema import StructuredDecisionFormat, StructuredValueFormat

if TYPE_CHECKING:
    from litellm import Choices, ModelResponse, Usage
    from litellm.types.utils import (  # pyright: ignore[reportMissingTypeStubs]
        ChatCompletionMessageCustomToolCall,
        ChatCompletionMessageToolCall,
    )

logger = logging.getLogger(__name__)


def handle_response(
    response: ModelResponse,
    *,
    requested_model: str,
    output: StructuredDecisionFormat | None,
    tool_argument_formats: dict[str, StructuredValueFormat] | None = None,
) -> LLMResponse:
    if not response.choices:
        raise LLMResponseDecodingError(
            LLMResponse(model=response.model),
            f"LLM returned empty choices (model={requested_model}). "
            "This may indicate a content filter, provider error, or a LiteLLM bug.",
        )

    choice: Choices = response.choices[0]
    message = choice.message
    usage = cast("Usage | None", cast(dict[str, Any], response).get("usage"))
    result = LLMResponse(
        model=response.model,
        content=message.content,
        reasoning_content=getattr(message, "reasoning_content", None),
        usage=usage.model_dump() if usage else None,
        stop_reason=choice.finish_reason,
        cost=_calculate_cost(response),
    )
    try:
        argument_formats = tool_argument_formats or {}
        result.tool_calls = [
            _function_tool_call(call, argument_formats)
            for call in (message.tool_calls or [])
        ]
    except (RuntimeError, ValueError) as error:
        raise LLMResponseDecodingError(result, str(error)) from error
    _decode_output(result, output)
    return result


def _function_tool_call(
    call: ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall,
    tool_argument_formats: dict[str, StructuredValueFormat],
) -> ToolCall:
    if getattr(call, "type", None) == "custom":
        raise RuntimeError("LLM returned an unsupported custom tool call")
    function_call = cast("ChatCompletionMessageToolCall", call)
    name = function_call.function.name
    if not isinstance(name, str) or not name:
        raise ValueError("Native tool call has no function name.")
    arguments_json = cast(object, function_call.function.arguments)
    if not isinstance(arguments_json, str):
        raise ValueError(f"Native tool call {name!r} has no JSON arguments.")
    arguments = LLMOutput.parse_json(arguments_json)
    value_format = tool_argument_formats.get(name)
    if value_format is not None:
        arguments = value_format.decode(arguments)
    return ToolCall(id=function_call.id, name=name, arguments=arguments)


def _decode_output(
    response: LLMResponse, output: StructuredDecisionFormat | None
) -> None:
    if output is None or response.content is None:
        return
    try:
        response.structured_output = output.decode_json(response.content)
    except ValueError:
        return


def _calculate_cost(response: ModelResponse) -> float | None:
    from litellm import cost_per_token

    usage = cast("Usage | None", cast(dict[str, Any], response).get("usage"))
    model = response.model
    if not (usage and model):
        return None
    try:
        prompt_cost, completion_cost = cost_per_token(
            model=model,
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
        )
        return prompt_cost + completion_cost
    except Exception:
        logger.warning("Failed to calculate cost for model %s", model, exc_info=True)
        return None


__all__ = ["handle_response"]
