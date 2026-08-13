from sefios import SessionScope
from sefios.cli import SefiaCLI, SefiaCLISession
from sefios.fastapi import SefiaHTTP, SefiaHTTPSession
from sefios.sessions import SessionManager


def test_public_lifecycle_facades_are_final():
    assert getattr(SessionScope, "__final__", False)
    assert getattr(SefiaCLI, "__final__", False)
    assert getattr(SefiaCLISession, "__final__", False)
    assert getattr(SefiaHTTP, "__final__", False)
    assert getattr(SefiaHTTPSession, "__final__", False)
    assert getattr(SessionManager, "__final__", False)
