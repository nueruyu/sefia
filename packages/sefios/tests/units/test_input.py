import pytest
from sefia.testing import MockLLMClient
from sefios import SessionScope, require_input
from sefios.exceptions import InteractionRequired
from sefios.interactions import InteractionChannel


async def test_application_input_in_plain_session_scope() -> None:
    scope = SessionScope(llm_client=MockLLMClient([]))
    with pytest.raises(InteractionRequired) as pause:
        async with scope.session(session_id="input"):
            await require_input("Continue?")
    assert pause.value.request == {"type": "input", "prompt": "Continue?"}
    async with scope.session(session_id="input"):
        await InteractionChannel(
            scope.persistence.create_session_storage("input")
        ).resolve(pause.value.interaction_id, "yes")
        assert await require_input("Continue?") == "yes"


def test_input_supports_only_prompt_preview_observation() -> None:
    from inspect import signature

    from sefios.tools import Input

    assert set(signature(Input).parameters) == {"on_prompt_delta"}
