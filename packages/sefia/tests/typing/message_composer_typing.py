from typing import Any, assert_type, cast

from sefia.inference import FunctionInfo
from sefia.llm import Message, MessageComposer, MessagePlan, TaskPrompt
from typing_extensions import override


class CustomMessages(MessageComposer):
    @override
    async def compose(self, function: FunctionInfo, plan: MessagePlan) -> MessagePlan:
        assert_type(function.prompt_arguments, dict[str, Any])
        return MessagePlan(
            parts=(Message(role="developer", content="instructions"), *plan.parts)
        )


def check_message_composer() -> None:
    composers = cast(tuple[MessageComposer, ...], (CustomMessages(),))
    assert_type(composers, tuple[MessageComposer, ...])
    assert_type(MessagePlan(parts=(TaskPrompt(arguments={}),)), MessagePlan)
