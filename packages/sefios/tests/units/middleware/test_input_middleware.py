from sefia import ToolRegistry
from sefia.inference import (
    StepDecision,
    ResultDecision,
    ToolCallsDecision,
    ToolCallRequest,
)
from sefia.testing import make_step_context, make_tool_call_request
from sefios.middleware import InputCallComposer
from sefios.tools import Input

HUMAN_INPUT_TOOL_NAME = "ask_human"


def _human_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.add(
        Input().get_input,
        name=HUMAN_INPUT_TOOL_NAME,
    )
    return registry


def _human_call(id_: str, prompt: str) -> ToolCallRequest:
    return make_tool_call_request(
        id=id_,
        name=HUMAN_INPUT_TOOL_NAME,
        arguments={"prompt": prompt},
    )


def _tool_call(id_: str, name: str = "search") -> ToolCallRequest:
    return make_tool_call_request(id=id_, name=name, arguments={"query": "sefia"})


async def _run(
    middleware: InputCallComposer, decision: StepDecision, step: int = 0
) -> StepDecision:
    async def nxt() -> StepDecision:
        return decision

    return await middleware.wrap(
        make_step_context(
            step=step,
            tool_registry=_human_registry(),
        ),
        nxt,
    )


class TestInputCallComposer:
    async def test_non_tool_decision_is_unchanged(self):
        middleware = InputCallComposer()
        decision = ResultDecision(result="done")

        assert await _run(middleware, decision) is decision

    async def test_tool_decision_without_input_calls_is_unchanged(self):
        middleware = InputCallComposer()
        decision = ToolCallsDecision(calls=[_tool_call("t1")])

        assert await _run(middleware, decision) is decision

    async def test_single_input_call_is_unchanged(self):
        middleware = InputCallComposer()
        decision = ToolCallsDecision(calls=[_human_call("h1", "What is the audience?")])

        assert await _run(middleware, decision) is decision

    async def test_multiple_input_calls_are_composed(self):
        decision = ToolCallsDecision(
            calls=[
                _human_call("h1", "What is the target audience?"),
                _human_call("h2", "What is the goal of the article?"),
            ]
        )

        composed = await _run(InputCallComposer(), decision)

        assert isinstance(composed, ToolCallsDecision)
        assert composed is not decision
        assert composed.calls == [
            make_tool_call_request(
                id="h1",
                name=HUMAN_INPUT_TOOL_NAME,
                arguments={
                    "prompt": (
                        "What is the target audience?\nWhat is the goal of the article?"
                    )
                },
            )
        ]

    async def test_non_human_tool_calls_are_preserved(self):
        before = _tool_call("t1", name="lookup")
        between = _tool_call("t2", name="calculate")
        after = _tool_call("t3", name="save")
        decision = ToolCallsDecision(
            calls=[
                before,
                _human_call("h1", "First prompt?"),
                between,
                _human_call("h2", "Second prompt?"),
                after,
            ]
        )

        composed = await _run(InputCallComposer(), decision)

        assert isinstance(composed, ToolCallsDecision)
        assert composed.calls[0] is before
        assert composed.calls[2] is between
        assert composed.calls[3] is after
        assert composed.calls[1] == make_tool_call_request(
            id="h1",
            name=HUMAN_INPUT_TOOL_NAME,
            arguments={
                "prompt": "First prompt?\nSecond prompt?",
            },
        )

    async def test_custom_prompt_composer_can_be_async(self):
        async def compose_prompts(prompts: list[str]) -> str:
            return " / ".join(prompts)

        decision = ToolCallsDecision(
            calls=[
                _human_call("h1", "First prompt?"),
                _human_call("h2", "Second prompt?"),
            ]
        )

        composed = await _run(
            InputCallComposer(compose_prompts=compose_prompts),
            decision,
        )

        assert isinstance(composed, ToolCallsDecision)
        assert composed.calls[0].arguments["prompt"] == (
            "First prompt? / Second prompt?"
        )

    async def test_sequential_input_steps_are_not_collapsed(self):
        middleware = InputCallComposer()
        first = ToolCallsDecision(calls=[_human_call("h1", "Who is the audience?")])
        second = ToolCallsDecision(
            calls=[_human_call("h2", "Educational or promotional?")]
        )

        assert await _run(middleware, first, step=0) is first
        assert await _run(middleware, second, step=1) is second

    async def test_invalid_input_arguments_are_left_unchanged(self):
        decision = ToolCallsDecision(
            calls=[
                make_tool_call_request(
                    id="h1",
                    name=HUMAN_INPUT_TOOL_NAME,
                    arguments={"prompt": "First?"},
                ),
                make_tool_call_request(
                    id="h2",
                    name=HUMAN_INPUT_TOOL_NAME,
                    arguments={"other": "Second?"},
                ),
            ]
        )

        assert await _run(InputCallComposer(), decision) is decision

    async def test_unregistered_input_name_is_left_unchanged(self):
        decision = ToolCallsDecision(
            calls=[
                make_tool_call_request(
                    id="h1",
                    name="Input_get_input",
                    arguments={"prompt": "First?"},
                ),
                make_tool_call_request(
                    id="h2",
                    name="Input_get_input",
                    arguments={"prompt": "Second?"},
                ),
            ]
        )

        async def nxt():
            return decision

        assert (
            await InputCallComposer().wrap(
                make_step_context(),
                nxt,
            )
            is decision
        )
