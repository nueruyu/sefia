import inspect

from sefia.llm import DecisionPrompt, PromptRenderer


class _RenderOnlyPromptRenderer(PromptRenderer):
    def render(self, prompt: DecisionPrompt) -> str:
        return prompt.function.instructions


def test_prompt_renderer_requires_tool_result_rendering() -> None:
    assert inspect.isabstract(_RenderOnlyPromptRenderer)
