from collections.abc import Sequence

from ._interfaces.history_store import HistoryStore
from .inference import HistoryItem


class TransientHistoryStore(HistoryStore):
    """
    The default, stateless :class:`HistoryStore`: it stores nothing.

    ``load`` always returns an empty history and ``save`` is a no-op, so the
    executor behaves exactly as without a store — history is rebuilt on resume
    by replaying the run's engraved steps. Use a persistent implementation
    (e.g. sefios' ``DurableHistoryStore``) when history must survive
    independently of the execution log, such as for compaction.
    """

    async def load(self) -> list[HistoryItem]:
        return []

    async def save(self, items: Sequence[HistoryItem]) -> None:
        pass
