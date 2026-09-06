import pytest
from sefia_fastapi.exceptions import UnknownSessionError as HTTPUnknownSessionError

from sefios import MemoryPersistence
from sefios.fastapi import SefiaHTTP
from sefios.tools import Input, Output


@pytest.fixture
def http() -> SefiaHTTP:
    return SefiaHTTP(model="gpt-4o-mini")


def test_input_tool_is_exposed(http: SefiaHTTP) -> None:
    assert isinstance(http.input_tool, Input)


def test_output_tool_is_exposed(http: SefiaHTTP) -> None:
    assert isinstance(http.output_tool, Output)


def test_created_session_is_known(http: SefiaHTTP) -> None:
    http.ensure_session(http.create_session())


def test_created_session_is_known_to_another_instance() -> None:
    persistence = MemoryPersistence()
    first = SefiaHTTP(model="gpt-4o-mini", persistence=persistence)
    second = SefiaHTTP(model="gpt-4o-mini", persistence=persistence)

    second.ensure_session(first.create_session())


def test_default_persistence_is_process_local() -> None:
    first = SefiaHTTP(model="gpt-4o-mini")
    second = SefiaHTTP(model="gpt-4o-mini")

    session_id = first.create_session()

    with pytest.raises(HTTPUnknownSessionError):
        second.ensure_session(session_id)


def test_unknown_session_raises_http_error(http: SefiaHTTP) -> None:
    with pytest.raises(HTTPUnknownSessionError) as exc_info:
        http.ensure_session("ghost")

    assert exc_info.value.session_id == "ghost"


def test_events_for_unknown_session_raises(http: SefiaHTTP) -> None:
    with pytest.raises(HTTPUnknownSessionError):
        http.events("ghost")


def test_events_for_known_session_returns_sse_response(http: SefiaHTTP) -> None:
    response = http.events(http.create_session())

    assert response.media_type == "text/event-stream"


async def test_session_for_unknown_session_raises(http: SefiaHTTP) -> None:
    with pytest.raises(HTTPUnknownSessionError):
        async with http.session(session_id="ghost"):
            pass
