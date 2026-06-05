import pytest

from sefia.events import NextTurnRequested
from sefia.handlers.max_turns import MaxTurnsHandler, RequestNextTurn


class TestMaxTurnsHandler:
    def _request(self, completed_turns: int) -> NextTurnRequested:
        return NextTurnRequested(completed_turns=completed_turns, history=[])

    async def test_permits_next_turn_below_limit(self):
        handler = MaxTurnsHandler(max_turns=3)

        # Turns 1 and 2 are already done; another turn is still within the limit.
        for completed in (1, 2):
            with pytest.raises(RequestNextTurn):
                await handler.handle(self._request(completed))

    async def test_stays_silent_at_limit(self):
        handler = MaxTurnsHandler(max_turns=3)

        # Three turns done: the handler must not permit a fourth.
        await handler.handle(self._request(completed_turns=3))  # Should not raise

    async def test_default_limit_is_single_turn(self):
        handler = MaxTurnsHandler()

        # One turn done and the default limit is 1, so no further turn is granted.
        await handler.handle(self._request(completed_turns=1))  # Should not raise

    async def test_none_means_unlimited(self):
        handler = MaxTurnsHandler(max_turns=None)

        with pytest.raises(RequestNextTurn):
            await handler.handle(self._request(completed_turns=1000))

    def test_event_types(self):
        handler = MaxTurnsHandler()
        assert handler.event_types == (NextTurnRequested,)
