from typing import TYPE_CHECKING

from sefia import preview
from sefia.streaming import ArgStream


async def standalone(question: str, count: int = 0) -> str:
    return f"{question}:{count}"


@preview(standalone)
async def _standalone_stream(tool_call_id: str, events: ArgStream) -> None:
    async for _ in events:
        pass


_ = _standalone_stream


class Toolkit:
    async def ask(self, question: str) -> str:
        return question

    @preview(ask)
    async def _ask_stream(self, tool_call_id: str, events: ArgStream) -> None:
        async for _ in events:
            pass

    @staticmethod
    async def static_ask(question: str) -> str:
        return question

    @staticmethod
    @preview(static_ask)
    async def _static_ask_stream(tool_call_id: str, events: ArgStream) -> None:
        async for _ in events:
            pass

    @classmethod
    async def class_ask(cls, question: str) -> str:
        return question

    @preview(class_ask)
    async def _class_ask_stream(cls, tool_call_id: str, events: ArgStream) -> None:
        async for _ in events:
            pass


if TYPE_CHECKING:
    toolkit = Toolkit()

    # preview is purely metadata: it never changes the decorated method's
    # own type, so ordinary calls through the class stay unaffected.
    standalone_result = standalone("q")
    method_result = toolkit.ask("q")
    staticmethod_result = Toolkit.static_ask("q")
    classmethod_result = Toolkit.class_ask("q")
