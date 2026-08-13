from sefia.exceptions import InferenceError, PauseException
from sefia_litellm.exceptions import (
    InferenceConnectionError,
    InferenceRateLimitError,
    InferenceTemporarilyUnavailableError,
    InferenceTimeoutError,
)

_TRANSIENT_ERRORS = (
    InferenceTimeoutError,
    InferenceConnectionError,
    InferenceRateLimitError,
    InferenceTemporarilyUnavailableError,
)


def test_transient_errors_are_recoverable_inference_errors():
    for exc in _TRANSIENT_ERRORS:
        assert issubclass(exc, InferenceError)
        assert issubclass(exc, PauseException)
        instance = exc("boom")
        assert isinstance(instance, InferenceError)
        assert isinstance(instance, PauseException)
