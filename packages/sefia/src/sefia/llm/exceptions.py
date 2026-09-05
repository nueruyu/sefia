from ._messages import LLMCompletion


class LLMCompletionDecodingError(ValueError):
    """The provider returned data the client could not represent safely.

    ``LLMClient`` implementations must raise this exception, rather than a generic
    decoding exception, when received data is malformed or cannot be mapped to an
    ``LLMCompletion``. The partial completion lets the inference strategy route the
    failure through its repair flow.
    """

    def __init__(self, completion: LLMCompletion, reason: str) -> None:
        super().__init__(reason)
        self.completion = completion


class DecisionDecodingError(ValueError):
    """A transport received a completion that did not encode a decision."""

    def __init__(self, completion: LLMCompletion, reason: str) -> None:
        super().__init__(reason)
        self.completion = completion


__all__ = ["DecisionDecodingError", "LLMCompletionDecodingError"]
