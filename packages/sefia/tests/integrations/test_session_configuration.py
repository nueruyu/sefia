import json
from dataclasses import dataclass
from unittest.mock import Mock

import glyff
import sefia

from typing_extensions import override

from sefia.inference import FunctionInfo
from sefia.llm import (
    LLMCompletion,
    Message,
    MessageComposer,
    MessagePlan,
    PromptRenderer,
)
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
    renderer.render_decision_instructions.return_value = "custom control"

    async with memory_session(client, prompt_renderer=renderer):
        report = await _Agent().generate_report(topic="custom")

    assert report == _Report("custom", "rendered")
    assert client.requests[0]["messages"] == [
        {"role": "user", "content": "custom prompt\n\ncustom control"}
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


class _ProfileMessage(MessageComposer):
    @override
    async def compose(self, function: FunctionInfo, plan: MessagePlan) -> MessagePlan:
        return MessagePlan(
            parts=(Message(role="developer", content="shared"), *plan.parts)
        )


async def test_profiles_share_session_message_composers() -> None:
    @infer
    @sefia.profile("alternate")
    async def answer(topic: str) -> str:
        """Answer the task."""
        ...

    default_client = MockLLMClient([])
    profile_client = MockLLMClient([result_completion("done")])
    async with memory_session(
        default_client,
        profiles=[sefia.Profile(key="alternate", client=profile_client)],
        message_composers=(_ProfileMessage(),),
    ):
        assert await answer("topic") == "done"

    assert not default_client.requests
    assert profile_client.requests[0]["messages"][0] == {
        "role": "developer",
        "content": "shared",
    }
