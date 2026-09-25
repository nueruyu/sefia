import inspect

from sefia.llm import InferencePrompt, PromptRenderer


class _RenderOnlyPromptRenderer(PromptRenderer):
    def render(self, prompt: InferencePrompt) -> str:
        return prompt.function.instructions


def test_prompt_renderer_requires_only_inference_rendering() -> None:
    assert not inspect.isabstract(_RenderOnlyPromptRenderer)
