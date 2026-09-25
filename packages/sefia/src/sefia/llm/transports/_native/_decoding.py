from ..._messages import ToolCall
from ...json import JsonSnapshot
from ...step_decision import StepTool


def decode_native_tool_calls(
    calls: list[ToolCall],
    result_tool: StepTool | None,
) -> JsonSnapshot:
    if not calls:
        raise ValueError("LLM did not call a native decision tool.")

    result_name = result_tool.name if result_tool is not None else None
    result_calls = [call for call in calls if call.name == result_name]
    if result_calls:
        if len(calls) != 1:
            raise ValueError("The result tool cannot be combined with other calls.")
        arguments = result_calls[0].arguments.as_object("result tool arguments")
        if set(arguments) != {"result"}:
            raise ValueError("The result tool requires exactly the 'result' field.")
        return JsonSnapshot.from_object(
            {
                "decision": JsonSnapshot.capture("result"),
                "result": arguments["result"],
            }
        )

    return JsonSnapshot.from_object(
        {
            "decision": JsonSnapshot.capture("tool_calls"),
            "tool_calls": JsonSnapshot.from_array(
                JsonSnapshot.from_object(
                    {
                        "name": JsonSnapshot.capture(call.name),
                        "arguments": JsonSnapshot.from_object(
                            call.arguments.as_object("tool arguments")
                        ),
                    }
                )
                for call in calls
            ),
        }
    )
