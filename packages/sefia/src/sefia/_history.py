from collections.abc import Sequence

import glyff

from ._interfaces.history_storage import HistorySnapshot, HistoryStorage
from .inference import HistoryItem

_METADATA_KEY = "sefia:history"


class GlyffHistoryStorage(HistoryStorage):
    """
    The default :class:`HistoryStorage`: the history lives in the run
    execution's glyff metadata, so no extra backend is needed and every run is
    durable out of the box.

    ``save`` writes the snapshot inside its own glyff transaction scope, so it
    commits immediately even though the surrounding ``@infer`` run has not
    finished (a long-lived, ``Never``-returning run never would). A resumed
    invocation re-enters the same run execution and ``load`` reads the snapshot
    back, so completed steps are not replayed. The snapshot is keyed on the run
    execution itself, so nested ``@infer`` runs each keep their own history.
    """

    async def load(self) -> HistorySnapshot:
        ctx = glyff.get_context()
        snapshot = await ctx.metadata.get(
            _METADATA_KEY,
            HistorySnapshot,
            execution_id=ctx.current_execution_id,
        )
        return snapshot if snapshot is not None else HistorySnapshot()

    async def save(self, snapshot: HistorySnapshot) -> None:
        ctx = glyff.get_context()
        # Own transaction scope: the surrounding run's scope only commits when
        # the run completes, which for a Never-returning chat is never. A nested
        # scope commits the metadata to the store immediately, so the snapshot
        # survives a pause.
        async with ctx.get_transaction_scope():
            await ctx.metadata.set(_METADATA_KEY, snapshot, HistorySnapshot)


class HistoryStore:
    """
    A run's live history and its persistence — the higher-level service over a
    :class:`HistoryStorage` (mirroring how sefios' ``StateStore`` sits over
    ``SessionStorage``).

    The executor owns the writes; middleware sees this object through
    :attr:`~sefia.StepContext.history` as a read-only view plus a single
    :meth:`rewrite` operation, so the raw storage and the in-place mutation stay
    encapsulated. Reads (:attr:`items`) hand back an immutable copy.
    """

    def __init__(self, storage: HistoryStorage):
        self._storage = storage
        self._items: list[HistoryItem] = []
        self._completed_steps = 0

    # --- middleware-facing surface ---

    @property
    def items(self) -> Sequence[HistoryItem]:
        """The current history as an immutable snapshot."""
        return tuple(self._items)

    @property
    def completed_steps(self) -> int:
        """How many steps of this run have finished."""
        return self._completed_steps

    async def rewrite(self, items: Sequence[HistoryItem]) -> None:
        """
        Replace the history with ``items`` (e.g. compaction).

        The new snapshot is persisted *before* the in-memory list is swapped,
        so a crash between the two leaves the store ahead (harmless) rather than
        behind (a lost rewrite). ``completed_steps`` is unchanged — a rewrite
        reshapes the conversation, it does not advance the run.
        """
        new_items = list(items)
        await self._storage.save(
            HistorySnapshot(tuple(new_items), self._completed_steps)
        )
        self._items[:] = new_items

    # --- executor-facing surface (same-package internal) ---

    async def _load(self) -> None:
        snapshot = await self._storage.load()
        self._items = list(snapshot.items)
        self._completed_steps = snapshot.completed_steps

    def _current(self) -> list[HistoryItem]:
        """The live list, passed to the engraved step (its content is the key)."""
        return self._items

    async def _record_step(
        self, decision: HistoryItem, results: Sequence[HistoryItem]
    ) -> None:
        """Append a completed step's decision and tool results, then persist.

        Called only after the step's engraved calls have committed, so a crash
        before this point resumes from the previous snapshot and the engraved
        decision/tool records replay the missing step.
        """
        self._items.append(decision)
        self._items.extend(results)
        self._completed_steps += 1
        await self._storage.save(
            HistorySnapshot(tuple(self._items), self._completed_steps)
        )
