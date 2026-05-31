import contextvars
from dataclasses import dataclass, field
from typing import Type, TypeVar

from glyff import Session as GlyffSession
from glyff.context import get_context as get_glyff_context
from pydantic import BaseModel

from .interfaces import InferenceStrategy, Policy, ToolCollector
from .interfaces.session_store import SessionStore
from .llm.client import LLMClient
from .state_store import StateStore

T = TypeVar("T", bound=BaseModel)


@dataclass
class InferenceContext:
    """
    Holds the context for an ongoing sefia inference session.
    """

    glyff_session: GlyffSession
    session_store: SessionStore
    llm_client: LLMClient
    inference_strategy: InferenceStrategy
    policies: list[Policy]
    tool_collector: ToolCollector
    _state_stores: dict[str, StateStore] = field(default_factory=dict)

    def get_call_state_store(
        self, key_suffix: str, state_type: Type[T]
    ) -> StateStore[T]:
        """
        Gets a StateStore scoped to the current engraved function call.
        This provides a private, persistent state for a single invocation
        of an engraved tool.
        """
        try:
            glyff_ctx = get_glyff_context()
            current_execution_id = glyff_ctx.current_execution_id
        except RuntimeError:
            current_execution_id = None

        if current_execution_id is None:
            raise RuntimeError(
                "get_call_state_store can only be used inside a @glyff.engrave function."
            )

        scoped_key = f"call_state::{str(current_execution_id)}::{key_suffix}"
        return self.get_state_store(scoped_key, state_type)

    def get_state_store(self, key: str, state_type: Type[T]) -> StateStore[T]:
        """
        Gets a StateStore for the given key and type, creating one if it doesn't exist.
        """
        if key not in self._state_stores:
            self._state_stores[key] = StateStore(
                store=self.session_store,
                key=key,
                state_type=state_type,
            )
        store = self._state_stores[key]
        if store._state_type != state_type:
            raise TypeError(
                f"State store for key '{key}' was already created with a different type."
            )
        return store


context_var = contextvars.ContextVar[InferenceContext]("sefia_context")


def get_context() -> InferenceContext:
    """Retrieves the current inference context."""
    try:
        return context_var.get()
    except LookupError:
        raise RuntimeError(
            "Inference context is not set. Are you running outside a sefia.Session?"
        )
