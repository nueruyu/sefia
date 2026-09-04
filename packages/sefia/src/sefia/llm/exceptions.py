from ._messages import LLMResponse


class LLMResponseDecodingError(ValueError):
    """The provider returned a response the client could not represent safely.

    ``LLMClient`` implementations must raise this exception, rather than a generic
    decoding exception, when a received response is malformed or cannot be mapped to
    ``LLMResponse``. The partial response lets the inference strategy route the
    failure through its repair flow.
    """

    def __init__(self, response: LLMResponse, reason: str) -> None:
        super().__init__(reason)
        self.response = response


__all__ = ["LLMResponseDecodingError"]
