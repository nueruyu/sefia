class SefiaError(Exception):
    """Base class for errors raised by sefia."""


class PauseException(SefiaError):
    """
    Base control-flow signal that pauses the current run so it can be resumed
    later.

    A tool (for example, one waiting on external human input) raises a subclass
    of this to interrupt the running inference and hand control back to the
    caller instead of returning a value. It is a *control* signal, not a
    failure: the executor lets it propagate untouched rather than stringifying it
    into the tool history or reporting it through ``InferenceFailed``.

    glyff records completed executions durably and leaves an interrupted one in
    its ``STARTED`` state, so re-invoking the workflow re-runs only the
    unfinished step and resumes from where it paused.

    Concrete pause signals (for example, a human-input pause defined by an
    application-facing layer) subclass this; the core only depends on the base
    contract.
    """


class ToolError(SefiaError):
    """Base class for errors raised by a tool."""


class ToolConflictError(SefiaError):
    """Raised when two tools with the same name are found."""


class InferenceError(PauseException):
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
    which the executor reports through ``InferenceFailed``.

    Adapters may define their own provider-shaped subclasses of this base (for
    example ``sefia_litellm`` defines ``InferenceTimeoutError`` and friends);
    the framework only depends on this abstract base.

    Because an ``InferenceError`` is recoverable, it also subclasses
    :class:`PauseException`. The executor therefore does **not** report it
    through ``InferenceFailed``; instead the run pauses and stays resumable, so
    re-invoking the workflow re-runs the step from scratch (and an in-loop
    ``Retrier`` may retry it within the same process first). The error object is
    preserved as it propagates, so callers can catch it either as an
    ``InferenceError`` (to inspect what went wrong) or as a ``PauseException``
    (to treat it as a pause).
    """


class InvalidInferenceResponseError(InferenceError):
    """
    The LLM produced a response that could not be parsed or validated against
    the expected schema.

    Treated as recoverable: LLM output is non-deterministic, so re-running the
    step (on resume, or via an in-loop retry) may yield a conforming response.
    ``LLMInferenceStrategy`` first retries in place with corrective feedback
    built from ``detail`` and ``raw_content``, so the model can repair its own
    output before the error ever propagates.

    ``detail`` describes what was wrong (the parse/validation error);
    ``raw_content`` carries the invalid response body when one was received.
    Both are kept as structured fields so feedback prompts and error reporting
    don't have to parse the exception message.
    """

    def __init__(self, detail: str, raw_content: str | None = None):
        message = detail if raw_content is None else f"{detail}, content: {raw_content}"
        super().__init__(message)
        self.detail = detail
        self.raw_content = raw_content


class UnknownToolDecisionError(ValueError):
    """Raised when an LLM decision calls a tool that is not available."""

    def __init__(self, tool_name: str):
        super().__init__(f"Unknown tool call: {tool_name}")
        self.tool_name = tool_name
