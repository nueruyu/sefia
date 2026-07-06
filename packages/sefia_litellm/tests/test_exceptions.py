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
    # Each adapter-defined transient error is an InferenceError, so the
    # framework treats it as recoverable, and (via InferenceError) a
    # PauseException, so glyff leaves the step resumable instead of engraving it.
    for exc in _TRANSIENT_ERRORS:
        assert issubclass(exc, InferenceError)
        assert issubclass(exc, PauseException)
        instance = exc("boom")
        assert isinstance(instance, InferenceError)
        assert isinstance(instance, PauseException)
