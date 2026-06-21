from sefia.exceptions import (
    ConnectionException,
    InferenceException,
    RateLimitException,
    SefiaError,
    TemporarilyUnavailableException,
    TimeoutException,
    ToolConflictError,
    ToolError,
)


def test_core_exceptions_share_sefia_base():
    assert issubclass(ToolError, SefiaError)
    assert issubclass(ToolConflictError, SefiaError)
    assert issubclass(InferenceException, SefiaError)


def test_inference_exceptions_share_inference_base():
    assert issubclass(TimeoutException, InferenceException)
    assert issubclass(ConnectionException, InferenceException)
    assert issubclass(RateLimitException, InferenceException)
    assert issubclass(TemporarilyUnavailableException, InferenceException)
