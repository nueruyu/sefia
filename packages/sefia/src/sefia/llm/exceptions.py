class RecoverableInferenceError(Exception):
    """
    Signals a transient, retryable failure during an inference step.

    LLM client adapters should raise this (mapping provider-specific transient
    errors such as timeouts, rate limits, and 5xx responses) instead of letting
    the raw provider exception propagate. The inference executor converts it into
    a ``glyff.exceptions.YieldException`` so the engraved step is interrupted
    gracefully and left resumable, rather than being engraved as a permanent
    ``FAILED`` record. This prevents a momentary outage from being persisted as
    an unrecoverable failure.
    """

    pass
