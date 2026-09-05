import json
from collections.abc import Callable

from sefia import Tools
from sefia.llm import LLMCompletion
from sefia.testing import MockLLMClient
from sefios import MemoryPersistence, SessionScope, domain
from sefios.tools import Output, OutputMessage

infer = domain("packages.sefios.tests.scenarios.test_output_tool", version="1").infer


class Agent:
    _output: Tools[Output]

    def __init__(self, output_tool: Output):
        self._output = output_tool

    @infer
    async def greet(self) -> str:
        """Send a greeting to the user, then report that it was sent."""
        ...


def _responses() -> list[LLMCompletion]:
    return [
        LLMCompletion(
            content=json.dumps(
                {
                    "decision": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "Output_send_output",
                            "arguments": {"message": "Hello there!"},
                        }
                    ],
                }
            )
        ),
        LLMCompletion(content=json.dumps({"decision": "result", "result": "sent"})),
    ]


class TestOutput:
    async def test_send_output_emits_once_and_replays_without_re_emitting(
        self,
        make_mock_llm: Callable[[list[LLMCompletion]], MockLLMClient],
    ) -> None:
        emitted: list[OutputMessage] = []
        agent = Agent(Output(on_output=emitted.append))
        session_id = "output-tool-test-1"
        persistence = MemoryPersistence()

        mock_llm = make_mock_llm(_responses())
        async with SessionScope(llm_client=mock_llm, persistence=persistence).session(
            session_id=session_id
        ):
            result = await agent.greet()

        assert result == "sent"
        assert [m.message for m in emitted] == ["Hello there!"]
        assert emitted[0].interaction_id

        # Re-invoking the same session must replay without re-emitting.
        replay_llm = make_mock_llm([])
        async with SessionScope(llm_client=replay_llm, persistence=persistence).session(
            session_id=session_id
        ):
            replayed = await agent.greet()

        assert replayed == "sent"
        assert len(replay_llm.requests) == 0
        assert [m.message for m in emitted] == ["Hello there!"]
