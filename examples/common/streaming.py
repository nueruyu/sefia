from sefia._interfaces import Policy
from sefia.event_system import EventHandler
from sefia.llm.events import LLMTokenReceived


class StreamingPrintHandler(EventHandler[LLMTokenReceived]):
    """An event handler that prints LLM tokens to the console."""

    @property
    def event_types(self):
        return (LLMTokenReceived,)

    async def handle(self, event: LLMTokenReceived):
        print(event.token, end="", flush=True)


class StreamingPolicy(Policy):
    """A policy that enables console streaming of LLM tokens."""

    def create_handlers(self) -> list[EventHandler]:
        return [StreamingPrintHandler()]
