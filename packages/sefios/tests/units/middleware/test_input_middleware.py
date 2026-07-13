from sefia import (
    HistorySnapshot,
    HistoryStorage,
    StepContext,
    ToolRegistry,
)
from sefia._history import StepHistory
from sefia.inference import ResultDecision, ToolCallDecision, ToolCallRequest
from sefios.middleware import InputCallComposer
from sefios.tools import InputTool

HUMAN_INPUT_TOOL_NAME = "ask_human"


class _NoHistory(HistoryStorage):
    async def load(self) -> HistorySnapshot:
        return HistorySnapshot()

    async def save(self, snapshot: HistorySnapshot) -> None:
        pass


def _empty_history() -> StepHistory:
    return StepHistory(_NoHistory())


def _human_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.add(
        InputTool().get_input,
        name=HUMAN_INPUT_TOOL_NAME,
    )
    return registry


def _human_call(id_: str, prompt: str) -> ToolCallRequest:
    return ToolCallRequest(
        id=id_,
        name=HUMAN_INPUT_TOOL_NAME,
        arguments={"prompt": prompt},
    )


def _tool_call(id_: str, name: str = "search") -> ToolCallRequest:
    return ToolCallRequest(id=id_, name=name, arguments={"query": "sefia"})


async def _run(middleware: InputCallComposer, decision, step: int = 0):
    async def nxt():
        return decision

    return await middleware.wrap(
        StepContext(
            step=step,
            history=_empty_history(),
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
        decision = ToolCallDecision(calls=[_tool_call("t1")])

        assert await _run(middleware, decision) is decision

    async def test_single_input_call_is_unchanged(self):
        middleware = InputCallComposer()
        decision = ToolCallDecision(calls=[_human_call("h1", "What is the audience?")])

        assert await _run(middleware, decision) is decision

    async def test_multiple_input_calls_are_composed(self):
        decision = ToolCallDecision(
            calls=[
                _human_call("h1", "What is the target audience?"),
                _human_call("h2", "What is the goal of the article?"),
            ]
        )

        composed = await _run(InputCallComposer(), decision)

        assert isinstance(composed, ToolCallDecision)
        assert composed is not decision
        assert composed.calls == [
            ToolCallRequest(
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
        decision = ToolCallDecision(
            calls=[
                before,
                _human_call("h1", "First prompt?"),
                between,
                _human_call("h2", "Second prompt?"),
                after,
            ]
        )

        composed = await _run(InputCallComposer(), decision)

        assert isinstance(composed, ToolCallDecision)
        assert composed.calls[0] is before
        assert composed.calls[2] is between
        assert composed.calls[3] is after
        assert composed.calls[1] == ToolCallRequest(
            id="h1",
            name=HUMAN_INPUT_TOOL_NAME,
            arguments={
                "prompt": "First prompt?\nSecond prompt?",
            },
        )

    async def test_custom_prompt_composer_can_be_async(self):
        async def compose_prompts(prompts: list[str]) -> str:
            return " / ".join(prompts)

        decision = ToolCallDecision(
            calls=[
                _human_call("h1", "First prompt?"),
                _human_call("h2", "Second prompt?"),
            ]
        )

        composed = await _run(
            InputCallComposer(compose_prompts=compose_prompts),
            decision,
        )

        assert isinstance(composed, ToolCallDecision)
        assert composed.calls[0].arguments["prompt"] == (
            "First prompt? / Second prompt?"
        )

    async def test_sequential_input_steps_are_not_collapsed(self):
        middleware = InputCallComposer()
        first = ToolCallDecision(calls=[_human_call("h1", "Who is the audience?")])
        second = ToolCallDecision(
            calls=[_human_call("h2", "Educational or promotional?")]
        )

        assert await _run(middleware, first, step=0) is first
        assert await _run(middleware, second, step=1) is second

    async def test_invalid_input_arguments_are_left_unchanged(self):
        decision = ToolCallDecision(
            calls=[
                ToolCallRequest(
                    id="h1",
                    name=HUMAN_INPUT_TOOL_NAME,
                    arguments={"prompt": "First?"},
                ),
                ToolCallRequest(
                    id="h2",
                    name=HUMAN_INPUT_TOOL_NAME,
                    arguments={"other": "Second?"},
                ),
            ]
        )

        assert await _run(InputCallComposer(), decision) is decision

    async def test_unregistered_input_name_is_left_unchanged(self):
        decision = ToolCallDecision(
            calls=[
                ToolCallRequest(
                    id="h1",
                    name="InputTool_get_input",
                    arguments={"prompt": "First?"},
                ),
                ToolCallRequest(
                    id="h2",
                    name="InputTool_get_input",
                    arguments={"prompt": "Second?"},
                ),
            ]
        )

        async def nxt():
            return decision

        assert (
            await InputCallComposer().wrap(
                StepContext(step=0, history=_empty_history()),
                nxt,
            )
            is decision
        )
