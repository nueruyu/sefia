from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from sefia import Policy
from sefia.event_system import EventHandler
from sefia.llm import Message
from sefia.llm.events import BeforeLLMCall

_console = Console()


class PromptDumpHandler(EventHandler[BeforeLLMCall]):
    """An event handler that prints LLM prompts to the console for debugging."""

    ROLE_STYLES = {
        "system": "bold blue",
        "user": "bold green",
        "assistant": "bold cyan",
        "tool": "bold magenta",
    }

    @property
    def event_types(self):
        return (BeforeLLMCall,)

    def _format_message(self, msg: Message) -> Text:
        """Formats a single message with role-based color."""
        style = self.ROLE_STYLES.get(msg.role, "white")
        text = Text()
        text.append(f"[{msg.role.upper()}]", style=style)
        if msg.content:
            text.append(f"\n{msg.content}")
        if msg.tool_calls:
            text.append(f"\nTool Calls: {msg.tool_calls}")
        return text

    async def handle(self, event: BeforeLLMCall):
        formatted_messages = Text()
        for index, msg in enumerate(event.messages):
            if index:
                formatted_messages.append("\n\n")
            formatted_messages.append_text(self._format_message(msg))

        _console.print(
            Panel(
                formatted_messages,
                title="LLM PROMPT",
                border_style="yellow",
                expand=False,
            )
        )


class VerbosePolicy(Policy):
    """A policy that enables console dumping of LLM prompts for debugging."""

    def create_handlers(self) -> list[EventHandler]:
        return [PromptDumpHandler()]
