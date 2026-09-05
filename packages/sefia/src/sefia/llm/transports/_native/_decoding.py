from ..._messages import ToolCall
from ...structured_data import StructuredData
from ...step_decision import StepTool


def decode_native_tool_calls(
    calls: list[ToolCall],
    result_tool: StepTool | None,
) -> StructuredData:
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
        return StructuredData.from_object(
            {
                "decision": StructuredData.from_json("result"),
                "result": arguments["result"],
            }
        )

    return StructuredData.from_object(
        {
            "decision": StructuredData.from_json("tool_calls"),
            "tool_calls": StructuredData.from_array(
                StructuredData.from_object(
                    {
                        "name": StructuredData.from_json(call.name),
                        "arguments": StructuredData.from_object(
                            call.arguments.to_object("tool arguments")
                        ),
                    }
                )
                for call in calls
            ),
        }
    )
