from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from sefia.llm import LLMCompletion, ToolCall
from sefia.llm.exceptions import LLMCompletionDecodingError
from sefia.llm.structured_data import StructuredData

from ._schema import StructuredDecisionFormat
from ._schema._data_format import StructuredDataFormat

if TYPE_CHECKING:
    from litellm import Choices, ModelResponse, Usage
    from litellm.types.utils import (  # pyright: ignore[reportMissingTypeStubs]
        ChatCompletionMessageCustomToolCall,
        ChatCompletionMessageToolCall,
    )

logger = logging.getLogger(__name__)


def decode_completion(
    response: ModelResponse,
    *,
    requested_model: str,
    decision_format: StructuredDecisionFormat | None,
    tool_data_formats: dict[str, StructuredDataFormat] | None = None,
) -> LLMCompletion:
    if not response.choices:
        raise LLMCompletionDecodingError(
            LLMCompletion(model=response.model),
            f"LLM returned empty choices (model={requested_model}). "
            "This may indicate a content filter, provider error, or a LiteLLM bug.",
        )

    choice: Choices = response.choices[0]
    message = choice.message
    usage = cast("Usage | None", cast(dict[str, Any], response).get("usage"))
    completion = LLMCompletion(
        model=response.model,
        content=message.content,
        reasoning_content=getattr(message, "reasoning_content", None),
        usage=usage.model_dump() if usage else None,
        stop_reason=choice.finish_reason,
        cost=_calculate_cost(response),
    )
    try:
        argument_formats = tool_data_formats or {}
        completion.tool_calls = [
            _decode_tool_call(call, argument_formats)
            for call in (message.tool_calls or [])
        ]
    except ValueError as error:
        raise LLMCompletionDecodingError(completion, str(error)) from error
    _decode_structured_data(completion, decision_format)
    return completion


def _decode_tool_call(
    call: ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall,
    tool_data_formats: dict[str, StructuredDataFormat],
) -> ToolCall:
    if getattr(call, "type", None) == "custom":
        raise ValueError("LLM returned an unsupported custom tool call")
    function_call = cast("ChatCompletionMessageToolCall", call)
    name = function_call.function.name
    if not isinstance(name, str) or not name:
        raise ValueError("Native tool call has no function name.")
    arguments_json = cast(object, function_call.function.arguments)
    if not isinstance(arguments_json, str):
        raise ValueError(f"Native tool call {name!r} has no JSON arguments.")
    arguments = StructuredData.parse_json(arguments_json)
    data_format = tool_data_formats.get(name)
    if data_format is not None:
        arguments = data_format.decode(arguments)
    return ToolCall(id=function_call.id, name=name, arguments=arguments)


def _decode_structured_data(
    completion: LLMCompletion,
    decision_format: StructuredDecisionFormat | None,
) -> None:
    if decision_format is None or completion.content is None:
        return
    try:
        completion.structured_output = decision_format.decode_json(completion.content)
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


__all__ = ["decode_completion"]
