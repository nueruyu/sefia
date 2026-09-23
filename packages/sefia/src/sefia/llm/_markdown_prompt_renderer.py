from typing_extensions import final, override

from ._prompt_renderer import InferencePrompt, PromptRenderer
from ._text import compact_json, json_block
from .step_decision import StepTool


@final
class MarkdownPromptRenderer(PromptRenderer):
    """Renders the standard inference prompt with Markdown and JSON content."""

    @override
    def render(self, prompt: InferencePrompt) -> str:
        sections = [f"# Task\n\n{prompt.function.instructions}"]
        sections.append(self._render_arguments(prompt))
        if prompt.tools:
            sections.append(self._render_tools(prompt.tools))
        return "\n\n".join(sections)

    def _render_arguments(self, prompt: InferencePrompt) -> str:
        arguments = prompt.arguments.to_json_value()
        if arguments == {}:
            return "## Task arguments\n\nNone."
        return "## Task arguments\n\n" + json_block(arguments)

    def _render_tools(self, definitions: tuple[StepTool, ...]) -> str:
        tools = "\n".join(self._render_tool(tool) for tool in definitions)
        return f"## Available tools\n\n{tools}"

    def _render_tool(self, tool: StepTool) -> str:
        description = f" — {tool.description}" if tool.description else ""
        schema = tool.arguments.to_dict()
        return f"- `{tool.name}`{description}\n  Arguments: {compact_json(schema)}"
