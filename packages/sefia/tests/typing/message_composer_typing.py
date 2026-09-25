from typing import Any, assert_type, cast

from sefia.inference import FunctionInfo
from sefia.llm import Message, MessageComposer, MessageLayout, StructuredData
from typing_extensions import override


class CustomMessages(MessageComposer):
    @override
    async def compose(
        self, function: FunctionInfo, layout: MessageLayout
    ) -> MessageLayout:
        assert_type(function.prompt_arguments, dict[str, Any])
        return MessageLayout(
            before=(Message(role="developer", content="instructions"), *layout.before),
            arguments=layout.arguments,
            after=layout.after,
        )


def check_message_composer() -> None:
    composers = cast(tuple[MessageComposer, ...], (CustomMessages(),))
    assert_type(composers, tuple[MessageComposer, ...])
    assert_type(MessageLayout(before=(), arguments={}, after=()), MessageLayout)
    assert_type(StructuredData.from_object({}), StructuredData)
