from sefia.event_system import EventHandler
from sefia.llm.events import LLMTokenReceived


class StreamingPrintHandler(EventHandler[LLMTokenReceived]):
    """Prints LLM tokens to the console as they arrive."""

    async def handle(self, event: LLMTokenReceived):
        print(event.token, end="", flush=True)
