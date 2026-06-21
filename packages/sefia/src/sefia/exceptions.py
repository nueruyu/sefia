from glyff.exceptions import YieldException


class SefiaError(Exception):
    """Base class for errors raised by sefia."""


class ToolError(SefiaError):
    """Base class for errors raised by a tool."""


class ToolConflictError(SefiaError):
    """Raised when two tools with the same name are found."""


class InferenceError(SefiaError, YieldException):
    """
    Base class for *recoverable* errors raised while performing an inference
    step.

    Client adapters translate provider-specific failures into ``InferenceError``
    so the rest of sefia never has to know about a particular provider's
    exception types. Only failures that can plausibly succeed on a later attempt
    are mapped here — a transient network/provider hiccup, or an LLM response
    that did not conform to the expected schema. Genuinely permanent failures
    (authentication, malformed request, content policy, ...) are deliberately
    *not* mapped to ``InferenceError`` and propagate as their own exceptions,
    which glyff engraves as genuine failures.

    Adapters may define their own provider-shaped subclasses of this base (for
    example ``sefia_litellm`` defines ``InferenceTimeoutError`` and friends);
    the framework only depends on this abstract base.

    Because an ``InferenceError`` is recoverable, it also subclasses glyff's
    ``YieldException``. glyff therefore does **not** engrave it as a permanent
    ``FAILED`` record; instead the step is left resumable, so re-invoking the
    workflow re-runs the step from scratch (and an in-loop ``Retrier`` may retry
    it within the same process first). The error object is preserved as it
    propagates, so callers can catch it either as an ``InferenceError`` (to
    inspect what went wrong) or as a ``YieldException`` (to treat it as a pause).
    """


class InvalidInferenceResponseError(InferenceError):
    """
    The LLM produced a response that could not be parsed or validated against
    the expected schema.

    Treated as recoverable: LLM output is non-deterministic, so re-running the
    step (on resume, or via an in-loop retry) may yield a conforming response.
    """
