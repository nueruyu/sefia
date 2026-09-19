import pytest

from sefios._input import InputRequest, InputResult, request_input
from sefios.exceptions import InputRequired


async def test_request_input_preserves_caller_identity_when_pending() -> None:
    request = InputRequest(interaction_id="caller-id", prompt="Continue?")
    pending: list[InputRequest] = []

    with pytest.raises(InputRequired) as pause:
        await request_input(request, lambda _: None, pending.append)

    assert pending == [request]
    assert pause.value.interaction_id == "caller-id"
    assert pause.value.prompt == "Continue?"


async def test_request_input_reports_completion() -> None:
    request = InputRequest(interaction_id="caller-id", prompt="Continue?")
    completed: list[InputResult] = []

    value = await request_input(
        request,
        lambda _: "yes",
        on_complete=completed.append,
    )

    assert value == "yes"
    assert completed == [
        InputResult(interaction_id="caller-id", prompt="Continue?", value="yes")
    ]
