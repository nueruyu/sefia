from sefia.llm import DecisionPrompt, PromptRenderer


class _RenderOnlyPromptRenderer(PromptRenderer):
    def render(self, prompt: DecisionPrompt) -> str:
        return prompt.response_instructions


def test_prompt_renderer_has_default_tool_result_rendering() -> None:
    renderer = _RenderOnlyPromptRenderer()

    assert renderer.render_tool_result("result") == "result"
