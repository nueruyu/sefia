from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Literal, get_args

import glyff
from typing_extensions import override

from sefia import (
    Domain,
    DecisionContext,
    DecisionMiddleware,
    InferenceContext,
    InferenceMiddleware,
    MessageContext,
    MessageMiddleware,
    MessagePlan,
    MiddlewareSet,
    Policy,
    Profile,
    TaskPrompt,
    Tools,
    policy,
    profile,
)
from sefia.llm import Message
from sefia.inference import StepDecision
from sefia.testing import (
    MockLLMClient,
    memory_session,
    result_completion,
    tool_calls_completion,
)


@dataclass(frozen=True)
class AsMessage:
    role: Literal["developer", "user"]


@dataclass(frozen=True)
class AsConversation:
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


class ConversationMessages(MessageMiddleware):
    @override
    async def wrap(
        self, ctx: MessageContext, nxt: Callable[[], Awaitable[MessagePlan]]
    ) -> MessagePlan:
        assert ctx.step == 0
        plan = await nxt()
        task = next(part for part in plan.parts if isinstance(part, TaskPrompt))
        remaining = dict(task.arguments)
        developer: list[Message] = []
        conversation: list[Message] = []
        current: list[Message] = []
        for name, hint in ctx.function.type_hints.items():
            if name not in remaining:
                continue
            metadata = get_args(hint)[1:]
            value = remaining[name]
            if marker := next((m for m in metadata if isinstance(m, AsMessage)), None):
                target = developer if marker.role == "developer" else current
                target.append(Message(role=marker.role, content=value))
                del remaining[name]
            elif any(isinstance(m, AsConversation) for m in metadata):
                conversation.extend(
                    Message(role=item.role, content=item.content) for item in value
                )
                del remaining[name]
        return MessagePlan(
            parts=(*developer, TaskPrompt(remaining), *conversation, *current)
        )


async def test_application_defined_annotations_compose_messages() -> None:
    infer = Domain(glyff.Domain("tests.message-app", version="1")).infer

    @infer
    @policy(Policy(middleware=lambda: MiddlewareSet(message=(ConversationMessages(),))))
    async def respond(
        instructions: Annotated[str, AsMessage("developer")],
        history: Annotated[list[ChatMessage], AsConversation()],
        message: Annotated[str, AsMessage("user")],
        knowledge: dict[str, str],
    ) -> str:
        """Continue the conversation."""
        ...

    client = MockLLMClient([result_completion("done")])
    async with memory_session(client, session_id="message-app"):
        assert (
            await respond(
                "Reply briefly in Japanese.",
                [
                    ChatMessage("user", "私はXXXです"),
                    ChatMessage("assistant", "よろしく"),
                ],
                "私の名前は？",
                {"foo": "bar"},
            )
            == "done"
        )

    messages = client.requests[0]["messages"]
    assert [message["role"] for message in messages] == [
        "developer",
        "user",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0]["content"] == "Reply briefly in Japanese."
    assert '"knowledge"' in messages[1]["content"]
    assert '"instructions"' not in messages[1]["content"]
    assert '"history"' not in messages[1]["content"]
    assert messages[-1]["content"] == "私の名前は？"


class _RunLayer(InferenceMiddleware):
    def __init__(self, label: str, log: list[str]) -> None:
        self.label = label
        self.log = log

    @override
    async def wrap(
        self, ctx: InferenceContext, nxt: Callable[[], Awaitable[object]]
    ) -> object:
        self.log.append(f"run:{self.label}:enter")
        result = await nxt()
        self.log.append(f"run:{self.label}:exit")
        return result


class _MessageLayer(MessageMiddleware):
    def __init__(self, label: str, log: list[str]) -> None:
        self.label = label
        self.log = log

    @override
    async def wrap(
        self, ctx: MessageContext, nxt: Callable[[], Awaitable[MessagePlan]]
    ) -> MessagePlan:
        self.log.append(f"message:{self.label}:enter")
        plan = await nxt()
        self.log.append(f"message:{self.label}:exit")
        return MessagePlan(
            parts=(Message(role="developer", content=self.label), *plan.parts)
        )


async def test_policy_factories_run_once_and_preserve_category_precedence() -> None:
    log: list[str] = []
    calls: dict[str, int] = {}

    def labeled_policy(label: str) -> Policy:
        def make() -> MiddlewareSet:
            calls[label] = calls.get(label, 0) + 1
            return MiddlewareSet(
                inference=(_RunLayer(label, log),),
                message=(_MessageLayer(label, log),),
            )

        return Policy(middleware=make)

    domain = Domain(
        glyff.Domain("tests.message-policy-order", version="1"),
        policies=[labeled_policy("domain")],
    )

    @domain.infer
    @policy(labeled_policy("function"))
    @profile("selected")
    async def answer(topic: str) -> str:
        """Answer the task."""
        ...

    client = MockLLMClient([result_completion("done")])
    async with memory_session(
        client,
        session_id="message-policy-order",
        policies=[labeled_policy("session")],
        profiles=[
            Profile(key="selected", client=client, policies=[labeled_policy("profile")])
        ],
    ):
        assert await answer("topic") == "done"

    labels = ["session", "domain", "profile", "function"]
    assert calls == dict.fromkeys(labels, 1)
    assert log == [
        *(f"run:{label}:enter" for label in labels),
        *(f"message:{label}:enter" for label in labels),
        *(f"message:{label}:exit" for label in reversed(labels)),
        *(f"run:{label}:exit" for label in reversed(labels)),
    ]
    assert [
        message["content"] for message in client.requests[0]["messages"][:4]
    ] == labels


class _RetryDecision(DecisionMiddleware):
    @override
    async def wrap(
        self, ctx: DecisionContext, nxt: Callable[[], Awaitable[StepDecision]]
    ) -> StepDecision:
        await nxt()
        return await nxt()


class _CountMessages(MessageMiddleware):
    def __init__(self) -> None:
        self.calls = 0

    @override
    async def wrap(
        self, ctx: MessageContext, nxt: Callable[[], Awaitable[MessagePlan]]
    ) -> MessagePlan:
        self.calls += 1
        plan = await nxt()
        return MessagePlan(
            parts=(Message(role="developer", content=str(self.calls)), *plan.parts)
        )


async def test_decision_middleware_retry_reruns_message_composition() -> None:
    messages = _CountMessages()
    infer = Domain(glyff.Domain("tests.message-retry", version="1")).infer

    @infer
    @policy(
        Policy(
            middleware=lambda: MiddlewareSet(
                decision=(_RetryDecision(),), message=(messages,)
            )
        )
    )
    async def answer(topic: str) -> str:
        """Answer the task."""
        ...

    client = MockLLMClient([result_completion("first"), result_completion("second")])
    async with memory_session(client, session_id="message-retry"):
        assert await answer("topic") == "second"

    assert messages.calls == 2
    assert [request["messages"][0]["content"] for request in client.requests] == [
        "1",
        "2",
    ]


class _RecordSteps(MessageMiddleware):
    def __init__(self, steps: list[int]) -> None:
        self.steps = steps

    @override
    async def wrap(
        self, ctx: MessageContext, nxt: Callable[[], Awaitable[MessagePlan]]
    ) -> MessagePlan:
        self.steps.append(ctx.step)
        return await nxt()


class _Lookup:
    async def lookup(self) -> str:
        return "found"


async def test_message_context_tracks_each_decision_step() -> None:
    steps: list[int] = []
    infer = Domain(glyff.Domain("tests.message-steps", version="1")).infer

    class Agent:
        tool: Tools[_Lookup]

        def __init__(self) -> None:
            self.tool = _Lookup()

        @infer
        @policy(
            Policy(middleware=lambda: MiddlewareSet(message=(_RecordSteps(steps),)))
        )
        async def answer(self) -> str:
            """Answer using the tool."""
            ...

    client = MockLLMClient(
        [tool_calls_completion(("_Lookup_lookup", {})), result_completion("done")]
    )
    async with memory_session(client, session_id="message-steps"):
        assert await Agent().answer() == "done"

    assert steps == [0, 1]
