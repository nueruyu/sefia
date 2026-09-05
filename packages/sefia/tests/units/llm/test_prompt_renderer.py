from sefia.llm import DecisionPrompt, PromptRenderer


class _RenderOnlyPromptRenderer(PromptRenderer):
    def render(self, prompt: DecisionPrompt) -> str:
        return prompt.response_instructions

    def render_tool_result(self, result: object) -> str:
        return str(result)


def test_prompt_renderer_defines_tool_result_rendering() -> None:
    renderer = _RenderOnlyPromptRenderer()

    assert renderer.render_tool_result("result") == "result"
