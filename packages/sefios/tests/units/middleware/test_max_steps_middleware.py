import pytest
from sefia.inference import ResultDecision
from sefia.testing import make_step_context
from sefios.middleware import MaxStepsExceededError, StepLimiter


async def _decision():
    return ResultDecision(result="done")


class TestStepLimiter:
    def test_rejects_non_positive_max_steps(self):
        with pytest.raises(ValueError):
            StepLimiter(max_steps=0)

    def test_allows_none_max_steps(self):
        StepLimiter(max_steps=None)

    async def test_runs_step_within_limit(self):
        middleware = StepLimiter(max_steps=3)

        for step in (0, 1, 2):
            decision = await middleware.wrap(make_step_context(step=step), _decision)
            assert isinstance(decision, ResultDecision)

    async def test_raises_once_limit_is_reached(self):
        middleware = StepLimiter(max_steps=3)

        with pytest.raises(MaxStepsExceededError):
            await middleware.wrap(make_step_context(step=3), _decision)

    async def test_does_not_call_next_when_over_limit(self):
        middleware = StepLimiter(max_steps=1)
        called = False

        async def nxt():
            nonlocal called
            called = True
            return ResultDecision(result="done")

        with pytest.raises(MaxStepsExceededError):
            await middleware.wrap(make_step_context(step=1), nxt)
        assert called is False

    async def test_none_means_unlimited(self):
        middleware = StepLimiter(max_steps=None)

        decision = await middleware.wrap(make_step_context(step=1000), _decision)
        assert isinstance(decision, ResultDecision)
