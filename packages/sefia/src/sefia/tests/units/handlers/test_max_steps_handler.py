import pytest

from sefia.events import StepStarted
from sefia.handlers.max_steps import MaxStepsExceededError, MaxStepsHandler


class TestMaxStepsHandler:
    def _step(self, step: int) -> StepStarted:
        return StepStarted(step=step, history=[])

    async def test_does_not_raise_within_limit(self):
        handler = MaxStepsHandler(max_steps=3)

        # Steps 0, 1, 2 are the three permitted steps.
        for step in (0, 1, 2):
            await handler.handle(self._step(step))  # Should not raise

    async def test_raises_once_limit_is_reached(self):
        handler = MaxStepsHandler(max_steps=3)

        # Step index 3 would be the fourth step, past the limit of 3.
        with pytest.raises(MaxStepsExceededError):
            await handler.handle(self._step(3))

    async def test_none_means_unlimited(self):
        handler = MaxStepsHandler(max_steps=None)

        await handler.handle(self._step(1000))  # Should not raise

    def test_event_types(self):
        handler = MaxStepsHandler(max_steps=1)
        assert handler.event_types == (StepStarted,)
