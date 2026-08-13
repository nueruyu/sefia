"""A :class:`~sefia.HistoryStorage` backed by the session's storage."""

from glyff import get_context as get_glyff_context
from glyff.exceptions import ContextNotSetError
from sefia import HistorySnapshot, HistoryStorage
from typing_extensions import final, override

from .._session_state import _execution_id_scope_key, get_session_storage


@final
class SessionHistoryStorage(HistoryStorage):
    """Persists each run's history in the sefios session storage, keyed by the
    run's glyff ``ExecutionId`` — an alternative to the default
    :class:`~sefia.history_storages.GlyffHistoryStorage` for keeping history out
    of glyff's execution records.
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

    @override
    async def load(self) -> HistorySnapshot:
        stored = await get_session_storage().get(self._key(), HistorySnapshot)
        return stored if stored is not None else HistorySnapshot()

    @override
    async def save(self, snapshot: HistorySnapshot) -> None:
        await get_session_storage().set(self._key(), snapshot, HistorySnapshot)
