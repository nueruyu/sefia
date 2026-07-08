from glyff_pydantic import PydanticSerializer

from sefios import MemorySessionStorage, SessionScope, get_session_storage


async def test_session_storage_factory_overrides_default(tmp_path, make_mock_llm):
    captured: dict[str, MemorySessionStorage] = {}

    def factory(session_id: str) -> MemorySessionStorage:
        storage = MemorySessionStorage(serializer=PydanticSerializer())
        captured[session_id] = storage
        return storage

    scope = SessionScope(
        session_dir=tmp_path,
        llm_client=make_mock_llm([]),
        session_storage_factory=factory,
    )

    async with scope.session(session_id="custom-store"):
        assert get_session_storage() is captured["custom-store"]
