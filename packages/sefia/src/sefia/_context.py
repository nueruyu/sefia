import contextvars
import hashlib
import json
from dataclasses import dataclass, field
from typing import Type, TypeVar

from glyff import ExecutionId
from glyff import Session as GlyffSession
from glyff.context import get_context as get_glyff_context

from ._interfaces import InferenceStrategy, Policy
from ._interfaces.session_store import SessionStore
from ._state_store import StateStore
from .llm._client import LLMClient
from .tools import ToolCollector

T = TypeVar("T")


def _execution_id_to_data(execution_id: ExecutionId) -> dict[str, object]:
    parent_id = execution_id.parent_id
    return {
        "name": execution_id.name,
        "sequence": execution_id.sequence,
        "args_hash": execution_id.args_hash,
        "parent_id": _execution_id_to_data(parent_id) if parent_id else None,
    }


def _execution_id_scope_key(execution_id: ExecutionId) -> str:
    data = _execution_id_to_data(execution_id)
    stable_repr = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable_repr.encode("utf-8")).hexdigest()


@dataclass
class SessionContext:
    """Holds the context for an ongoing sefia inference session."""

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
        This provides call-local state for a single invocation of an engraved tool.
        """
        try:
            glyff_ctx = get_glyff_context()
            current_execution_id = glyff_ctx.current_execution_id
        except RuntimeError:
            current_execution_id = None

        if current_execution_id is None:
            raise RuntimeError(
                "get_call_state_store can only be used inside an engraved function."
            )

        scope_key = _execution_id_scope_key(current_execution_id)
        scoped_key = f"call_state/{scope_key}/{key_suffix}"
        return self.get_state_store(scoped_key, state_type)

    def get_state_store(self, key: str, state_type: Type[T]) -> StateStore[T]:
        """Gets a StateStore for the given key and type, creating one if needed."""
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


context_var = contextvars.ContextVar[SessionContext]("sefia_context")


def get_context() -> SessionContext:
    """Retrieves the current inference context."""
    try:
        return context_var.get()
    except LookupError:
        raise RuntimeError(
            "Inference context is not set. Are you running outside a sefia.Session?"
        )
