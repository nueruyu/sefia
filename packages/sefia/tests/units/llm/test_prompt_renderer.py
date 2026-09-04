import pytest

from sefia.llm import DecisionPrompt, PromptRenderer


class _RenderOnlyPromptRenderer(PromptRenderer):
    def render(self, prompt: DecisionPrompt) -> str:
        return prompt.response_instructions


def test_render_tool_result_is_optional_for_non_native_transports() -> None:
    renderer = _RenderOnlyPromptRenderer()

    with pytest.raises(NotImplementedError, match="NativeDecisionTransport"):
        renderer.render_tool_result("result")
