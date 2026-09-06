import json
from dataclasses import dataclass
from unittest.mock import Mock

import glyff
import sefia

from sefia.llm import LLMCompletion, PromptRenderer
from sefia.llm.transports import PromptedDecisionTransport
from sefia.testing import MockLLMClient, memory_session, result_completion

infer = sefia.Domain(
    glyff.Domain(
        "packages.sefia.tests.integrations.test_session_configuration", version="1"
    )
).infer


@dataclass
class _Report:
    topic: str
    summary: str


class _Agent:
    @infer
    async def generate_report(self, topic: str) -> _Report: ...


async def test_session_connects_a_custom_prompt_renderer_to_the_transport() -> None:
    client = MockLLMClient([result_completion(_Report("custom", "rendered"))])
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = "custom prompt"

    async with memory_session(client, prompt_renderer=renderer):
        report = await _Agent().generate_report(topic="custom")

    assert report == _Report("custom", "rendered")
    assert client.requests[0]["messages"] == [
        {"role": "user", "content": "custom prompt"}
    ]


async def test_session_connects_a_prompted_decision_transport() -> None:
    client = MockLLMClient(
        [
            LLMCompletion(
                content=json.dumps(
                    {
                        "decision": "result",
                        "result": {"topic": "prompted", "summary": "decoded"},
                    }
                )
            )
        ]
    )

    async with memory_session(client, decision_transport=PromptedDecisionTransport()):
        report = await _Agent().generate_report(topic="prompted")

    assert report == _Report("prompted", "decoded")
    assert client.requests[0]["decision_spec"] is None
