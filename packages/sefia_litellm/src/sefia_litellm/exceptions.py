"""Provider-shaped recoverable inference errors raised by the LiteLLM adapter.

These subclass :class:`sefia.exceptions.InferenceError` (and therefore
:class:`sefia.exceptions.PauseException`), so the rest of sefia treats them as
recoverable: a transient provider failure is never reported as a permanent
failure, and the run stays resumable. They live here, in the adapter, because the
exact set of transport
failure modes is a property of how this client maps a provider's exceptions; the
framework only needs the abstract ``InferenceError`` contract.
"""

from sefia.exceptions import InferenceError


class InferenceTimeoutError(InferenceError):
    """The inference request did not complete within the allotted time."""


class InferenceConnectionError(InferenceError):
    """The inference request could not reach the provider."""


class InferenceRateLimitError(InferenceError):
    """The request was rejected because a rate limit was exceeded."""


class InferenceTemporarilyUnavailableError(InferenceError):
    """The provider was temporarily unable to serve the request."""
