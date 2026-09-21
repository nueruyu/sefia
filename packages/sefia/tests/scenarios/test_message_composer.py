from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Literal, get_args

import glyff
from typing_extensions import override

from sefia import (
    DecisionContext,
    DecisionMiddleware,
    Domain,
    InferenceContext,
    InferenceMiddleware,
    MiddlewareSet,
    Policy,
    Profile,
    Tools,
    policy,
    profile,
)
from sefia.event_system import EventHandler
from sefia.inference import FunctionInfo, StepDecision
from sefia.llm import Message, MessageComposer, MessagePlan, TaskPrompt
from sefia.llm.events import BeforeLLMCall
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


class ConversationMessages(MessageComposer):
    @override
    async def compose(self, function: FunctionInfo, plan: MessagePlan) -> MessagePlan:
        task = next(part for part in plan.parts if isinstance(part, TaskPrompt))
        remaining = dict(task.arguments)
        developer: list[Message] = []
        conversation: list[Message] = []
        current: list[Message] = []
        for name, hint in function.type_hints.items():
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
    async def respond(
        instructions: Annotated[str, AsMessage("developer")],
        history: Annotated[list[ChatMessage], AsConversation()],
        message: Annotated[str, AsMessage("user")],
        knowledge: dict[str, str],
    ) -> str:
        """Continue the conversation."""
        ...

    client = MockLLMClient([result_completion("done")])
    async with memory_session(
        client,
        session_id="message-app",
        message_composers=(ConversationMessages(),),
    ):
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
    assert [message["role"] for message in messages[:5]] == [
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
    assert messages[4]["content"] == "私の名前は？"
    assert messages[-1]["content"].startswith("## Response")


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


class _DecisionLayer(DecisionMiddleware):
    def __init__(self, label: str, log: list[str]) -> None:
        self.label = label
        self.log = log

    @override
    async def wrap(
        self, ctx: DecisionContext, nxt: Callable[[], Awaitable[StepDecision]]
    ) -> StepDecision:
        self.log.append(f"decision:{self.label}:enter")
        result = await nxt()
        self.log.append(f"decision:{self.label}:exit")
        return result


async def test_policy_factories_run_once_and_preserve_category_precedence() -> None:
    log: list[str] = []
    calls: dict[str, int] = {}

    def labeled_policy(label: str) -> Policy:
        def make() -> MiddlewareSet:
            calls[label] = calls.get(label, 0) + 1
            return MiddlewareSet(
                inference=(_RunLayer(label, log),),
                decision=(_DecisionLayer(label, log),),
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
        *(f"decision:{label}:enter" for label in labels),
        *(f"decision:{label}:exit" for label in reversed(labels)),
        *(f"run:{label}:exit" for label in reversed(labels)),
    ]


class _RetryDecision(DecisionMiddleware):
    @override
    async def wrap(
        self, ctx: DecisionContext, nxt: Callable[[], Awaitable[StepDecision]]
    ) -> StepDecision:
        await nxt()
        return await nxt()


class _CountMessages(MessageComposer):
    def __init__(self) -> None:
        self.calls = 0

    @override
    async def compose(self, function: FunctionInfo, plan: MessagePlan) -> MessagePlan:
        self.calls += 1
        return MessagePlan(
            parts=(Message(role="developer", content=str(self.calls)), *plan.parts)
        )


async def test_decision_middleware_retry_reruns_message_composition() -> None:
    composer = _CountMessages()
    infer = Domain(glyff.Domain("tests.message-retry", version="1")).infer

    @infer
    @policy(Policy(middleware=lambda: MiddlewareSet(decision=(_RetryDecision(),))))
    async def answer(topic: str) -> str:
        """Answer the task."""
        ...

    client = MockLLMClient([result_completion("first"), result_completion("second")])
    async with memory_session(
        client, session_id="message-retry", message_composers=(composer,)
    ):
        assert await answer("topic") == "second"

    assert composer.calls == 2
    assert [request["messages"][0]["content"] for request in client.requests] == [
        "1",
        "2",
    ]


class _Lookup:
    async def lookup(self) -> str:
        return "found"


async def test_new_step_recomposes_application_plan() -> None:
    composer = _CountMessages()
    infer = Domain(glyff.Domain("tests.message-steps", version="1")).infer

    class Agent:
        tool: Tools[_Lookup]

        def __init__(self) -> None:
            self.tool = _Lookup()

        @infer
        async def answer(self) -> str:
            """Answer using the tool."""
            ...

    client = MockLLMClient(
        [tool_calls_completion(("_Lookup_lookup", {})), result_completion("done")]
    )
    async with memory_session(
        client, session_id="message-steps", message_composers=(composer,)
    ):
        assert await Agent().answer() == "done"

    assert composer.calls == 2
    assert [request["messages"][0]["content"] for request in client.requests] == [
        "1",
        "2",
    ]


class _FixedApplicationMessage(MessageComposer):
    def __init__(self, message: Message) -> None:
        self.message = message

    @override
    async def compose(self, function: FunctionInfo, plan: MessagePlan) -> MessagePlan:
        return MessagePlan(parts=(self.message, *plan.parts))


class _MutateObservedMessages(EventHandler[BeforeLLMCall]):
    @override
    async def handle(self, event: BeforeLLMCall) -> None:
        message = event.messages[0]
        message.role = "user"
        assert isinstance(message.content, list)
        message.content[0]["text"] = "changed by handler"
        event.messages[-1].content = "changed control"


async def test_before_llm_call_handler_cannot_change_client_request() -> None:
    application_message = Message(role="developer", content=[{"text": "original"}])
    infer = Domain(glyff.Domain("tests.message-observation", version="1")).infer

    @infer
    @policy(Policy(handlers=lambda: [_MutateObservedMessages()]))
    async def answer(topic: str) -> str:
        """Answer the task."""
        ...

    client = MockLLMClient([result_completion("done")])
    async with memory_session(
        client,
        session_id="message-observation",
        message_composers=(_FixedApplicationMessage(application_message),),
    ):
        assert await answer("topic") == "done"

    sent = client.requests[0]["messages"]
    assert sent[0] == {"role": "developer", "content": [{"text": "original"}]}
    assert sent[-1]["content"].startswith("## Response")
    assert application_message.content == [{"text": "original"}]
