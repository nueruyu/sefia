from sefia.llm import LLMClient, PromptRenderer
from sefia.llm.transports import (
    DecisionObserver,
    DecisionRequest,
    DecisionTransport,
    DecodedDecision,
)
from typing_extensions import override


class CustomDecisionTransport(DecisionTransport):
    @override
    async def request_decision(
        self,
        client: LLMClient,
        prompt_renderer: PromptRenderer,
        request: DecisionRequest,
        observer: DecisionObserver,
        stream: bool,
    ) -> DecodedDecision:
        raise NotImplementedError


_transport: DecisionTransport = CustomDecisionTransport()
