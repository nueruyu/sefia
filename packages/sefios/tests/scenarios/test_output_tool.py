import json

import glyff
from glyff import ArgsHasher, Serializer
from glyff.store import MemoryBackend

from sefia import Session, infer
from sefia.llm import LLMResponse
from sefios import MemorySessionStorage
from sefios._session_state import bind_session_storage
from sefios.tools import OutputMessage, OutputTool


class Agent:
    def __init__(self, output_tool: OutputTool):
        self._output = output_tool

    @infer
    async def greet(self) -> str:
        """Send a greeting to the user, then report that it was sent."""
        ...


def _responses() -> list[LLMResponse]:
    return [
        LLMResponse(
            content=json.dumps(
                {
                    "decision": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "OutputTool_send_output",
                            "arguments": {"message": "Hello there!"},
                        }
                    ],
                }
            )
        ),
        LLMResponse(content=json.dumps({"decision": "result", "result": "sent"})),
    ]


class TestOutputTool:
    async def test_send_output_emits_once_and_replays_without_re_emitting(
        self, serializer: Serializer, hasher: ArgsHasher, make_mock_llm
    ):
        emitted: list[OutputMessage] = []
        agent = Agent(OutputTool(on_output=emitted.append))
        session_id = "output-tool-test-1"
        glyff_store = MemoryBackend()
        sefia_store = MemorySessionStorage(serializer=serializer)

        mock_llm = make_mock_llm(_responses())
        async with glyff.Session(
            id=session_id, backend=glyff_store, serializer=serializer, hasher=hasher
        ) as gs:
            with bind_session_storage(sefia_store):
                async with Session(llm_client=mock_llm, glyff_session=gs):
                    result = await agent.greet()

        assert result == "sent"
        assert [m.message for m in emitted] == ["Hello there!"]
        assert emitted[0].interaction_id

        # Re-invoking the same session must replay without re-emitting.
        replay_llm = make_mock_llm([])
        async with glyff.Session(
            id=session_id, backend=glyff_store, serializer=serializer, hasher=hasher
        ) as gs:
            with bind_session_storage(sefia_store):
                async with Session(llm_client=replay_llm, glyff_session=gs):
                    replayed = await agent.greet()

        assert replayed == "sent"
        assert len(replay_llm.requests) == 0
        assert [m.message for m in emitted] == ["Hello there!"]
