from sefia import StepContext
from sefia.inference import ResultDecision, ToolCallDecision, ToolCallRequest
from sefios.middleware import ComposeHumanInputStepMiddleware, compose_human_input_calls
from sefios.tools import HUMAN_INPUT_TOOL_NAME


def _human_call(id_: str, question: str) -> ToolCallRequest:
    return ToolCallRequest(
        id=id_,
        name=HUMAN_INPUT_TOOL_NAME,
        arguments={"question": question},
    )


def _tool_call(id_: str, name: str = "search") -> ToolCallRequest:
    return ToolCallRequest(id=id_, name=name, arguments={"query": "sefia"})


async def _run(middleware: ComposeHumanInputStepMiddleware, decision, step: int = 0):
    async def nxt():
        return decision

    return await middleware.wrap(StepContext(step=step, history=[]), nxt)


class TestComposeHumanInputStepMiddleware:
    async def test_non_tool_decision_is_unchanged(self):
        middleware = ComposeHumanInputStepMiddleware()
        decision = ResultDecision(result="done")

        assert await _run(middleware, decision) is decision

    async def test_tool_decision_without_human_input_calls_is_unchanged(self):
        middleware = ComposeHumanInputStepMiddleware()
        decision = ToolCallDecision(calls=[_tool_call("t1")])

        assert await _run(middleware, decision) is decision

    async def test_single_human_input_call_is_unchanged(self):
        middleware = ComposeHumanInputStepMiddleware()
        decision = ToolCallDecision(calls=[_human_call("h1", "What is the audience?")])

        assert await _run(middleware, decision) is decision

    async def test_multiple_human_input_calls_are_composed(self):
        decision = ToolCallDecision(
            calls=[
                _human_call("h1", "What is the target audience?"),
                _human_call("h2", "What is the goal of the article?"),
            ]
        )

        composed = compose_human_input_calls(decision)

        assert composed is not decision
        assert composed.calls == [
            ToolCallRequest(
                id="h1",
                name=HUMAN_INPUT_TOOL_NAME,
                arguments={
                    "question": (
                        "Please answer these together:\n"
                        "1. What is the target audience?\n"
                        "2. What is the goal of the article?"
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
                _human_call("h1", "First question?"),
                between,
                _human_call("h2", "Second question?"),
                after,
            ]
        )

        composed = compose_human_input_calls(decision)

        assert composed.calls[0] is before
        assert composed.calls[2] is between
        assert composed.calls[3] is after
        assert composed.calls[1] == ToolCallRequest(
            id="h1",
            name=HUMAN_INPUT_TOOL_NAME,
            arguments={
                "question": (
                    "Please answer these together:\n"
                    "1. First question?\n"
                    "2. Second question?"
                )
            },
        )

    async def test_sequential_human_input_steps_are_not_collapsed(self):
        middleware = ComposeHumanInputStepMiddleware()
        first = ToolCallDecision(calls=[_human_call("h1", "Who is the audience?")])
        second = ToolCallDecision(
            calls=[_human_call("h2", "Educational or promotional?")]
        )

        assert await _run(middleware, first, step=0) is first
        assert await _run(middleware, second, step=1) is second

    async def test_invalid_human_input_arguments_are_left_unchanged(self):
        decision = ToolCallDecision(
            calls=[
                ToolCallRequest(
                    id="h1",
                    name=HUMAN_INPUT_TOOL_NAME,
                    arguments={"question": "First?"},
                ),
                ToolCallRequest(
                    id="h2",
                    name=HUMAN_INPUT_TOOL_NAME,
                    arguments={"other": "Second?"},
                ),
            ]
        )

        assert compose_human_input_calls(decision) is decision
