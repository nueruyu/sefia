class InferenceControlSignal(Exception):
    """
    Base class for the typed control signals that steer the executor's loops.

    These are not failures to be retried; they are deliberate instructions (or
    terminal limits) that the executor's loops interpret directly. A middleware
    raises one of these to communicate with the executor instead of relying on
    an exception escaping an event handler.
    """


class RequestInferenceRetry(InferenceControlSignal):
    """Signal asking the executor to discard the current run and start over."""


class MaxRetriesExceededError(InferenceControlSignal):
    """Raised when the configured number of retries has been exhausted."""


class MaxStepsExceededError(InferenceControlSignal):
    """Raised when an inference run exceeds its maximum number of steps."""


class StagnationError(InferenceControlSignal):
    """Raised when the inference run appears stuck repeating the same tool call."""
