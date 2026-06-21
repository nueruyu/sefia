from sefia.exceptions import (
    InferenceConnectionError,
    InferenceError,
    InferenceRateLimitError,
    InferenceTemporarilyUnavailableError,
    InferenceTimeoutError,
    SefiaError,
    ToolConflictError,
    ToolError,
)


def test_core_exceptions_share_sefia_base():
    assert issubclass(ToolError, SefiaError)
    assert issubclass(ToolConflictError, SefiaError)
    assert issubclass(InferenceError, SefiaError)


def test_inference_errors_share_inference_base():
    assert issubclass(InferenceTimeoutError, InferenceError)
    assert issubclass(InferenceConnectionError, InferenceError)
    assert issubclass(InferenceRateLimitError, InferenceError)
    assert issubclass(InferenceTemporarilyUnavailableError, InferenceError)
