"""The durable :class:`~sefia.HistoryStore` backed by the session's storage."""

from collections.abc import Sequence

from glyff import get_context as get_glyff_context
from glyff.exceptions import ContextNotSetError
from sefia import HistoryStore
from sefia.inference import HistoryItem

from ._session_state import _execution_id_scope_key, get_session_storage


class DurableHistoryStore(HistoryStore):
    """
    Persists each ``@infer`` run's history in the session's
    :class:`~sefios.SessionStorage`, keyed by the run's glyff ``ExecutionId``
    (the same scoping rule as :func:`~sefios.get_call_state_store`), so a
    resumed invocation re-enters the same run and loads the same history.

    Because the history lives outside glyff's engraved execution log, it can
    be rewritten — compacted — without disturbing replay: resuming loads the
    saved snapshot directly instead of rebuilding history by replaying every
    prior step. This is what makes an unbounded, ``Never``-returning chat run
    compactable (see ``sefios.middleware.HistoryCompactor``).

    The instance is stateless: storage and scope key are resolved per call
    from the bound session state and the current glyff context, so one
    instance can be shared across sessions and concurrent runs.
    """

    _KEY_PREFIX = "inference_history"

    def _key(self) -> str:
        try:
            glyff_ctx = get_glyff_context()
        except ContextNotSetError:
            glyff_ctx = None
        execution_id = (
            glyff_ctx.current_execution_id if glyff_ctx is not None else None
        )
        if execution_id is None:
            raise RuntimeError(
                "DurableHistoryStore can only be used inside an engraved "
                "inference run."
            )
        return f"{self._KEY_PREFIX}/{_execution_id_scope_key(execution_id)}"

    async def load(self) -> list[HistoryItem]:
        stored = await get_session_storage().get(self._key(), list[HistoryItem])
        return list(stored) if stored is not None else []

    async def save(self, items: Sequence[HistoryItem]) -> None:
        await get_session_storage().set(self._key(), list(items), list[HistoryItem])
