from sefia.llm import LLMCompletion, Message, ToolCall
from sefia.llm.structured_data import StructuredData
from sefia.testing import MockLLMClient


async def test_mock_llm_client_snapshots_core_messages() -> None:
    client = MockLLMClient([LLMCompletion(content="done")])

    await client.complete(
        [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="lookup",
                        arguments=StructuredData.from_json({"key": "item"}),
                    )
                ],
            )
        ]
    )

    assert client.requests[0]["messages"] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "lookup",
                    "arguments": {"key": "item"},
                }
            ],
        }
    ]
