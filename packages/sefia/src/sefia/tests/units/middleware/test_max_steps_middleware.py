import pytest

from sefia.interfaces.middleware import StepContext
from sefia.middleware.max_steps import MaxStepsMiddleware
from sefia.middleware.signals import MaxStepsExceededError
from sefia.models import FinalAnswerDecision


def _ctx(step: int) -> StepContext:
    return StepContext(step=step, history=[])


async def _decision():
    return FinalAnswerDecision(answer="done")


class TestMaxStepsMiddleware:
    async def test_runs_step_within_limit(self):
        middleware = MaxStepsMiddleware(max_steps=3)

        # Steps 0, 1, 2 are the three permitted steps.
        for step in (0, 1, 2):
            decision = await middleware.wrap(_ctx(step), _decision)
            assert isinstance(decision, FinalAnswerDecision)

    async def test_raises_once_limit_is_reached(self):
        middleware = MaxStepsMiddleware(max_steps=3)

        # Step index 3 would be the fourth step, past the limit of 3.
        with pytest.raises(MaxStepsExceededError):
            await middleware.wrap(_ctx(3), _decision)

    async def test_does_not_call_next_when_over_limit(self):
        middleware = MaxStepsMiddleware(max_steps=1)
        called = False

        async def nxt():
            nonlocal called
            called = True
            return FinalAnswerDecision(answer="done")

        with pytest.raises(MaxStepsExceededError):
            await middleware.wrap(_ctx(1), nxt)
        assert called is False

    async def test_none_means_unlimited(self):
        middleware = MaxStepsMiddleware(max_steps=None)

        decision = await middleware.wrap(_ctx(1000), _decision)
        assert isinstance(decision, FinalAnswerDecision)
