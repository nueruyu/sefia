from sefia.llm import Message, MessageLayout
from sefia.testing import make_function_info


def test_default_layout_keeps_prompt_arguments_between_empty_sides() -> None:
    function = make_function_info(bound_arguments={"topic": "sefia"})
    assert MessageLayout.default(function) == MessageLayout(
        before=(), arguments={"topic": "sefia"}, after=()
    )


def test_layout_allowsapplication_messages_around_inference_prompt() -> None:
    before = Message(role="developer", content="instructions")
    after = Message(role="user", content="question")
    layout = MessageLayout(before=(before,), arguments={}, after=(after,))
    assert layout.before == (before,)
    assert layout.arguments == {}
    assert layout.after == (after,)
