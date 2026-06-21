class SefiaError(Exception):
    """Base class for errors raised by sefia."""


class ToolError(SefiaError):
    """Base class for errors raised by a tool."""


class ToolConflictError(SefiaError):
    """Raised when two tools with the same name are found."""


class InferenceError(SefiaError):
    """
    Base class for errors raised by an LLM client while performing an inference
    step.

    Client adapters translate provider-specific failures into these abstract
    errors so the rest of sefia never has to know about a particular
    provider's exception types. sefia itself does not decide whether any of these
    are recoverable; the failure is published as an ``InferenceStepFailed`` event
    for observation, then engraved as a genuine failure. (Observation handlers
    cannot turn it into a resumable interrupt — the publisher isolates their
    exceptions — so resumability is driven by the control/execution layer.)
    """


class InferenceTimeoutError(InferenceError):
    """The inference request did not complete within the allotted time."""


class InferenceConnectionError(InferenceError):
    """The inference request could not reach the provider."""


class InferenceRateLimitError(InferenceError):
    """The request was rejected because a rate limit was exceeded."""


class InferenceTemporarilyUnavailableError(InferenceError):
    """The provider was temporarily unable to serve the request."""
