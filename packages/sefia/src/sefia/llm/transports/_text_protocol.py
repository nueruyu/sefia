from .._markdown import json_block
from .._messages import Message
from ..structured_data import StructuredData
from ._base import DecisionHistoryItem, DecisionToolCalls


def text_history_messages(
    history: tuple[DecisionHistoryItem, ...],
) -> list[Message]:
    if not history:
        return []

    records: list[StructuredData] = []
    for item in history:
        if isinstance(item, DecisionToolCalls):
            records.extend(
                StructuredData.from_object(
                    {
                        "tool_call": StructuredData.from_object(
                            {
                                "id": StructuredData.from_scalar(call.id),
                                "name": StructuredData.from_scalar(call.name),
                                "arguments": call.arguments,
                            }
                        )
                    }
                )
                for call in item.calls
            )
        else:
            records.append(
                StructuredData.from_object(
                    {
                        "tool_result": StructuredData.from_object(
                            {
                                "id": StructuredData.from_scalar(item.tool_call_id),
                                "result": item.result,
                            }
                        )
                    }
                )
            )
    data = StructuredData.from_array(records)
    return [
        Message(
            role="user",
            content=(
                "## Previous tool interactions\n\n" + json_block(data.to_json_value())
            ),
        )
    ]
