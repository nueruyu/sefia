from typing import TYPE_CHECKING

from sefia import tool
from sefia.streaming import ArgStream


@tool
async def standalone(question: str, count: int = 0) -> str:
    return f"{question}:{count}"


@standalone.stream
async def _standalone_stream(events: ArgStream) -> None:
    async for _ in events:
        pass


class Agent:
    @tool
    async def ask(self, question: str) -> str:
        return question

    @ask.stream
    async def _ask_stream(self, events: ArgStream) -> None:
        async for _ in events:
            pass

    @tool
    @staticmethod
    async def static_ask(question: str) -> str:
        return question

    @static_ask.stream
    async def _static_ask_stream(events: ArgStream) -> None:
        async for _ in events:
            pass

    @tool
    @classmethod
    async def class_ask(cls, question: str) -> str:
        return question

    @class_ask.stream
    async def _class_ask_stream(cls, events: ArgStream) -> None:
        async for _ in events:
            pass


if TYPE_CHECKING:
    agent = Agent()

    standalone_result = standalone("q")
    method_result = agent.ask("q")
    staticmethod_result = Agent.static_ask("q")
    classmethod_result = Agent.class_ask("q")
