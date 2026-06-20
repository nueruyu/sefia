from sefia import infer
from sefios.tools import HumanInputTool


class ChatAgent:
    def __init__(self, human_input: HumanInputTool):
        self._human_input = human_input

    @infer
    async def chat(self) -> str:
        """
        You are a helpful assistant having a conversation with a user.

        Loop using the HumanInputTool:
        1. Call HumanInputTool to get the user's message.
        2. Reply to it.
        3. Repeat from step 1.

        When the user says "exit", "quit", or "goodbye", stop and return
        a short farewell message.
        """
        ...
