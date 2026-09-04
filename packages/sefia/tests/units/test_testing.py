from sefia.llm import LLMOutput, LLMResponse, Message, ToolCall
from sefia.testing import MockLLMClient


async def test_mock_llm_client_snapshots_core_messages() -> None:
    client = MockLLMClient([LLMResponse(content="done")])

    await client.complete(
        [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="lookup",
                        arguments=LLMOutput.from_json({"key": "item"}),
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
