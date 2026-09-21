import pytest

from sefia.llm import Message, MessagePlan, TaskPrompt
from sefia.testing import make_function_info


def test_default_plan_contains_task_prompt() -> None:
    function = make_function_info(bound_arguments={"topic": "sefia"})
    assert MessagePlan.default(function) == MessagePlan(
        parts=(TaskPrompt(arguments={"topic": "sefia"}),)
    )


@pytest.mark.parametrize("parts", [(), (TaskPrompt({}), TaskPrompt({}))])
def test_plan_requires_exactly_one_task_prompt(parts: tuple[TaskPrompt, ...]) -> None:
    with pytest.raises(ValueError, match="exactly one TaskPrompt"):
        MessagePlan(parts=parts)


def test_plan_allows_application_messages_around_empty_task() -> None:
    first = Message(role="developer", content="instructions")
    last = Message(role="user", content="question")
    plan = MessagePlan(parts=(first, TaskPrompt(arguments={}), last))
    assert plan.parts == (first, TaskPrompt(arguments={}), last)
