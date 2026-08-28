from typing import Any

from rich.console import Console
from rich.panel import Panel
from sefia import Policy
from sefia.event_system import EventHandler
from sefia.llm.events import BeforeLLMCall

_console = Console()


class PromptDumpHandler(EventHandler[BeforeLLMCall]):
    """An event handler that prints LLM prompts to the console for debugging."""

    async def handle(self, event: BeforeLLMCall):
        _console.print(
            Panel(
                event.prompt,
                title="LLM PROMPT",
                border_style="yellow",
                expand=False,
            )
        )


class VerbosePolicy(Policy):
    """A policy that enables console dumping of LLM prompts for debugging."""

    def create_handlers(self) -> list[EventHandler[Any]]:
        return [PromptDumpHandler()]
