from typing import Self, Type, TypeVar

import glyff
from pydantic import BaseModel

from .context import InferenceContext, context_var
from .interfaces import (
    Policy,
    ToolCollector,
)
from .interfaces.session_store import SessionStore
from .llm.client import LLMClient
from .llm.strategy import LLMInferenceStrategy
from .policies import StagnationPolicy
from .state_store import StateStore
from .tool_collectors.collector import DefaultToolCollector

T = TypeVar("T", bound=BaseModel)


class Session:
    """
    Manages the lifecycle of an inference execution.
    Wraps a glyff.Session and sets up the sefia InferenceContext.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        glyff_session: glyff.Session,
        session_store: SessionStore,
        policies: list[Policy] | None = None,
        tool_collector: ToolCollector | None = None,
    ):
        self.llm_client = llm_client
        self._glyff_session = glyff_session
        self.session_store = session_store
        self._context_token = None
        self.policies: list[Policy] = [StagnationPolicy()] + (policies or [])
        self._tool_collector = tool_collector or DefaultToolCollector()
        self._inference_strategy = LLMInferenceStrategy(llm_client)
        self._context: InferenceContext | None = None

    def get_state_store(self, key: str, state_type: Type[T]) -> StateStore[T]:
        """
        Gets a StateStore for the given key and type, which can be used to
        manage persistent state within the session.
        """
        if self._context is None:
            raise RuntimeError(
                "Cannot get a state store before the session is entered."
            )
        return self._context.get_state_store(key, state_type)

    async def __aenter__(self) -> Self:
        self._context = InferenceContext(
            glyff_session=self._glyff_session,
            session_store=self.session_store,
            llm_client=self.llm_client,
            inference_strategy=self._inference_strategy,
            policies=self.policies,
            tool_collector=self._tool_collector,
        )
        self._context_token = context_var.set(self._context)
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        if self._context_token is not None:
            context_var.reset(self._context_token)
            self._context_token = None
