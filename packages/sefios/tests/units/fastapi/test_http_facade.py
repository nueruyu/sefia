import pytest
from sefia_fastapi import UnknownSessionError as HTTPUnknownSessionError
from sefios.fastapi import SefiaHTTP
from sefios.tools import Input


@pytest.fixture
def http(tmp_path) -> SefiaHTTP:
    return SefiaHTTP(session_dir=tmp_path / "sessions", model="gpt-4o-mini")


class TestSefiaHTTPSessionManagement:
    def test_input_tool_is_exposed(self, http: SefiaHTTP):
        assert isinstance(http.input_tool, Input)

    def test_created_session_is_known(self, http: SefiaHTTP):
        session_id = http.create_session()

        http.ensure_session(session_id)

    def test_unknown_session_raises_http_error(self, http: SefiaHTTP):
        # The facade raises the sefia_fastapi exception that applications map
        # to an HTTP response.
        with pytest.raises(HTTPUnknownSessionError) as exc_info:
            http.ensure_session("ghost")

        assert exc_info.value.session_id == "ghost"

    def test_events_for_unknown_session_raises(self, http: SefiaHTTP):
        with pytest.raises(HTTPUnknownSessionError):
            http.events("ghost")

    def test_events_for_known_session_returns_sse_response(self, http: SefiaHTTP):
        session_id = http.create_session()

        response = http.events(session_id)

        assert response.media_type == "text/event-stream"

    async def test_session_for_unknown_session_raises(self, http: SefiaHTTP):
        with pytest.raises(HTTPUnknownSessionError):
            async with http.session(session_id="ghost"):
                pass
