from abc import ABC, abstractmethod

from ..inference import FunctionInfo
from ._message_layout import MessageLayout


class MessageComposer(ABC):
    """Transforms an LLM message layout using application-defined conventions.

    Implementations should treat FunctionInfo as read-only and remain reentrant.
    Per-call data belongs in the function and layout, not mutable composer state.
    """

    @abstractmethod
    async def compose(
        self, function: FunctionInfo, layout: MessageLayout
    ) -> MessageLayout: ...
