from typing import TYPE_CHECKING

from sefia import stream_for
from sefia.streaming import ArgStream


async def standalone(question: str, count: int = 0) -> str:
    return f"{question}:{count}"


@stream_for(standalone)
async def _standalone_stream(events: ArgStream) -> None:
    async for _ in events:
        pass


class Toolkit:
    async def ask(self, question: str) -> str:
        return question

    @stream_for(ask)
    async def _ask_stream(self, events: ArgStream) -> None:
        async for _ in events:
            pass

    @staticmethod
    async def static_ask(question: str) -> str:
        return question

    @stream_for(static_ask)
    async def _static_ask_stream(events: ArgStream) -> None:
        async for _ in events:
            pass

    @classmethod
    async def class_ask(cls, question: str) -> str:
        return question

    @stream_for(class_ask)
    async def _class_ask_stream(cls, events: ArgStream) -> None:
        async for _ in events:
            pass


if TYPE_CHECKING:
    toolkit = Toolkit()

    # stream_for is purely metadata: it never changes the decorated method's
    # own type, so ordinary calls through the class stay unaffected.
    standalone_result = standalone("q")
    method_result = toolkit.ask("q")
    staticmethod_result = Toolkit.static_ask("q")
    classmethod_result = Toolkit.class_ask("q")
