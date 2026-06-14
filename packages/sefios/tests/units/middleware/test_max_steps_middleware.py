import pytest
from sefia import StepContext
from sefia.inference import FinalAnswerDecision
from sefios.core.middleware import MaxStepsExceededError, StepLimiter


def _ctx(step: int) -> StepContext:
    return StepContext(step=step, history=[])


async def _decision():
    return FinalAnswerDecision(answer="done")


class TestStepLimiter:
    def test_rejects_non_positive_max_steps(self):
        with pytest.raises(ValueError):
            StepLimiter(max_steps=0)

    def test_allows_none_max_steps(self):
        StepLimiter(max_steps=None)

    async def test_runs_step_within_limit(self):
        middleware = StepLimiter(max_steps=3)

        for step in (0, 1, 2):
            decision = await middleware.wrap(_ctx(step), _decision)
            assert isinstance(decision, FinalAnswerDecision)

    async def test_raises_once_limit_is_reached(self):
        middleware = StepLimiter(max_steps=3)

        with pytest.raises(MaxStepsExceededError):
            await middleware.wrap(_ctx(3), _decision)

    async def test_does_not_call_next_when_over_limit(self):
        middleware = StepLimiter(max_steps=1)
        called = False

        async def nxt():
            nonlocal called
            called = True
            return FinalAnswerDecision(answer="done")

        with pytest.raises(MaxStepsExceededError):
            await middleware.wrap(_ctx(1), nxt)
        assert called is False

    async def test_none_means_unlimited(self):
        middleware = StepLimiter(max_steps=None)

        decision = await middleware.wrap(_ctx(1000), _decision)
        assert isinstance(decision, FinalAnswerDecision)
