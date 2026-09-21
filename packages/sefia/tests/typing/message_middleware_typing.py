from collections.abc import Awaitable, Callable
from typing import Any, assert_type

from sefia import (
    MessageContext,
    MessageMiddleware,
    MessagePlan,
    MiddlewareSet,
    TaskPrompt,
)
from sefia.llm import Message
from typing_extensions import override


class CustomMessages(MessageMiddleware):
    @override
    async def wrap(
        self, ctx: MessageContext, nxt: Callable[[], Awaitable[MessagePlan]]
    ) -> MessagePlan:
        assert_type(ctx.step, int)
        assert_type(ctx.function.prompt_arguments, dict[str, Any])
        plan = await nxt()
        return MessagePlan(
            parts=(Message(role="developer", content="instructions"), *plan.parts)
        )


def check_message_policy() -> None:
    middleware = MiddlewareSet(message=(CustomMessages(),))
    assert_type(middleware.message, tuple[MessageMiddleware, ...])
    assert_type(MessagePlan(parts=(TaskPrompt(arguments={}),)), MessagePlan)
