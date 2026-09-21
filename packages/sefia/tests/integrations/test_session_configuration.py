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
    MessageLayout,
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

    async with memory_session(client, prompt_renderer=renderer):
        report = await _Agent().generate_report(topic="custom")

    assert report == _Report("custom", "rendered")
    messages = client.requests[0]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].startswith("custom prompt\n\n## Response\n\n")


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
    async def compose(
        self, function: FunctionInfo, layout: MessageLayout
    ) -> MessageLayout:
        return MessageLayout(
            before=(Message(role="developer", content="shared"), *layout.before),
            arguments=layout.arguments,
            after=layout.after,
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
