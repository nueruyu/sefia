import pytest

from sefia.events import AttemptStart, BeforeInferenceStep
from sefia.handlers.max_turns import MaxTurnsExceededError, MaxTurnsHandler


class TestMaxTurnsHandler:
    def _step(self) -> BeforeInferenceStep:
        return BeforeInferenceStep(history=[], tools=[])

    async def test_does_not_raise_within_limit(self):
        handler = MaxTurnsHandler(max_turns=3)

        for _ in range(3):
            await handler.handle(self._step())  # Should not raise

    async def test_raises_when_limit_exceeded(self):
        handler = MaxTurnsHandler(max_turns=3)

        for _ in range(3):
            await handler.handle(self._step())

        with pytest.raises(MaxTurnsExceededError):
            await handler.handle(self._step())

    async def test_attempt_start_resets_counter(self):
        handler = MaxTurnsHandler(max_turns=2)

        await handler.handle(self._step())
        await handler.handle(self._step())

        # A new attempt resets the counter, so the limit applies again afresh.
        await handler.handle(AttemptStart())

        await handler.handle(self._step())
        await handler.handle(self._step())  # Should not raise

        with pytest.raises(MaxTurnsExceededError):
            await handler.handle(self._step())

    def test_event_types(self):
        handler = MaxTurnsHandler()
        assert AttemptStart in handler.event_types
        assert BeforeInferenceStep in handler.event_types
