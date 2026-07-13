"""A :class:`~sefia.HistoryStorage` backed by the session's storage."""

from glyff import get_context as get_glyff_context
from glyff.exceptions import ContextNotSetError
from sefia import HistorySnapshot, HistoryStorage

from .._session_state import _execution_id_scope_key, get_session_storage


class SessionHistoryStorage(HistoryStorage):
    """
    Persists each ``@infer`` run's history in the session's
    :class:`~sefios.SessionStorage`, keyed by the run's glyff ``ExecutionId``
    (the same scoping rule as :func:`~sefios.get_call_state_store`).

    An alternative to the default :class:`~sefia.GlyffHistoryStorage`: use it to
    keep history in the sefios session storage (alongside application state,
    where it can be inspected on its own) rather than inside glyff's execution
    records. Both are durable and support compaction; this one just redirects
    where the snapshot lives.
    """

    _KEY_PREFIX = "inference_history"

    def _key(self) -> str:
        try:
            glyff_ctx = get_glyff_context()
        except ContextNotSetError:
            glyff_ctx = None
        execution_id = glyff_ctx.current_execution_id if glyff_ctx is not None else None
        if execution_id is None:
            raise RuntimeError(
                "SessionHistoryStorage can only be used inside an engraved "
                "inference run."
            )
        return f"{self._KEY_PREFIX}/{_execution_id_scope_key(execution_id)}"

    async def load(self) -> HistorySnapshot:
        stored = await get_session_storage().get(self._key(), HistorySnapshot)
        return stored if stored is not None else HistorySnapshot()

    async def save(self, snapshot: HistorySnapshot) -> None:
        await get_session_storage().set(self._key(), snapshot, HistorySnapshot)
