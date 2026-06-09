class ToolError(Exception):
    """Base class for errors raised by a tool."""


class FileOperationToolError(ToolError):
    """Base for file-related tool errors."""

    def __init__(self, message: str, path: str):
        super().__init__(message)
        self.path = path


class FileNotFoundToolError(FileOperationToolError):
    """Raised when a file is not found."""


class PermissionDeniedToolError(FileOperationToolError):
    """Raised when a file cannot be accessed."""


class InferenceException(Exception):
    """
    Base class for errors raised by an LLM client while performing an inference
    step.

    Client adapters translate provider-specific failures into these abstract
    exceptions so the rest of sefia never has to know about a particular
    provider's exception types. sefia itself does not decide whether any of these
    are recoverable; the failure is published as an ``InferenceStepFailed`` event
    for observation, then engraved as a genuine failure. (Observation handlers
    cannot turn it into a resumable interrupt — the publisher isolates their
    exceptions — so resumability is driven by the control/execution layer.)
    """


class TimeoutException(InferenceException):
    """The inference request did not complete within the allotted time."""


class ConnectionException(InferenceException):
    """The inference request could not reach the provider."""


class RateLimitException(InferenceException):
    """The request was rejected because a rate limit was exceeded."""


class TemporarilyUnavailableException(InferenceException):
    """The provider was temporarily unable to serve the request."""
