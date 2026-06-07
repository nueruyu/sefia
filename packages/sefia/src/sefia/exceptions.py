class InferenceException(Exception):
    """
    Base class for errors raised by an LLM client while performing an inference
    step.

    Client adapters translate provider-specific failures into these abstract
    exceptions so the rest of sefia (and consumer-supplied event handlers) never
    has to know about a particular provider's exception types. sefia itself does
    not decide whether any of these are recoverable; an ``InferenceStepFailed``
    event carries the exception, and a handler may choose to interrupt the
    session (by raising ``glyff.exceptions.YieldException``) or let it propagate.
    """


class TimeoutException(InferenceException):
    """The inference request did not complete within the allotted time."""


class ConnectionException(InferenceException):
    """The inference request could not reach the provider."""


class RateLimitException(InferenceException):
    """The request was rejected because a rate limit was exceeded."""


class TemporarilyUnavailableException(InferenceException):
    """The provider was temporarily unable to serve the request."""
