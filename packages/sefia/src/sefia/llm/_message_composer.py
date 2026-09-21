from abc import ABC, abstractmethod

from ..inference import FunctionInfo
from ._message_plan import MessagePlan


class MessageComposer(ABC):
    """Transforms an LLM message plan using application-defined conventions.

    Implementations should treat FunctionInfo as read-only and remain reentrant.
    Per-call data belongs in the function and plan, not mutable composer state.
    """

    @abstractmethod
    async def compose(
        self, function: FunctionInfo, plan: MessagePlan
    ) -> MessagePlan: ...
