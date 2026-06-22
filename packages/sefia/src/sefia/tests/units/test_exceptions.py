from glyff.exceptions import YieldException

from sefia.exceptions import (
    InferenceError,
    InvalidInferenceResponseError,
    SefiaError,
    ToolConflictError,
    ToolError,
)


def test_core_exceptions_share_sefia_base():
    assert issubclass(ToolError, SefiaError)
    assert issubclass(ToolConflictError, SefiaError)
    assert issubclass(InferenceError, SefiaError)


def test_inference_errors_share_inference_base():
    assert issubclass(InvalidInferenceResponseError, InferenceError)


def test_inference_errors_are_recoverable_yields():
    # An InferenceError is recoverable: it is also a YieldException, so glyff
    # treats it as a non-engraved, resumable interrupt rather than a permanent
    # failure. It remains a SefiaError so it is still catchable as one.
    assert issubclass(InferenceError, YieldException)
    assert issubclass(InferenceError, SefiaError)
    for exc in (InferenceError, InvalidInferenceResponseError):
        assert issubclass(exc, YieldException)
        instance = exc("boom")
        assert isinstance(instance, YieldException)
        assert isinstance(instance, InferenceError)
