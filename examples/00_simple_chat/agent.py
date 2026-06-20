from sefia import infer


class ChatAgent:
    @infer
    async def reply(self, message: str) -> str:
        """
        You are a helpful assistant having a conversation with a user.
        Reply to the user's message naturally and concisely.
        """
        ...
