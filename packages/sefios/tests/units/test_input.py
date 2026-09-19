import pytest
from sefia.testing import MockLLMClient
from sefios import SessionScope, require_input
from sefios._interaction_context import get_interaction_channel
from sefios.exceptions import InteractionRequired


async def test_application_input_in_plain_session_scope() -> None:
    scope = SessionScope(llm_client=MockLLMClient([]))
    with pytest.raises(InteractionRequired) as pause:
        async with scope.session(session_id="input"):
            await require_input("Continue?")
    assert pause.value.request == {"type": "input", "prompt": "Continue?"}
    async with scope.session(session_id="input"):
        await get_interaction_channel().resolve(pause.value.interaction_id, "yes")
        assert await require_input("Continue?") == "yes"
