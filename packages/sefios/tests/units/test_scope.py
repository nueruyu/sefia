from glyff_pydantic import PydanticSerializer

from sefios import MemorySessionStore, SessionScope, get_session_state


async def test_session_store_factory_overrides_default(tmp_path, make_mock_llm):
    captured: dict[str, MemorySessionStore] = {}

    def factory(session_id: str) -> MemorySessionStore:
        store = MemorySessionStore(serializer=PydanticSerializer())
        captured[session_id] = store
        return store

    scope = SessionScope(
        session_dir=tmp_path,
        llm_client=make_mock_llm([]),
        session_store_factory=factory,
    )

    async with scope.session(session_id="custom-store"):
        assert get_session_state().store is captured["custom-store"]
