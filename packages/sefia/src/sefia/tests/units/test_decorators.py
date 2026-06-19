import pytest

from sefia import InferenceContext, InferenceMiddleware, StepContext, StepMiddleware
from sefia._decorators import _partition_middleware
from sefia.inference import InferenceDecision


class _InferenceMiddlewareFixture(InferenceMiddleware):
    async def wrap(self, ctx: InferenceContext, nxt):
        return await nxt()


class _StepMiddlewareFixture(StepMiddleware):
    async def wrap(self, ctx: StepContext, nxt) -> InferenceDecision:
        return await nxt()


class TestPartitionMiddleware:
    def test_splits_by_seam(self):
        inference_middleware = _InferenceMiddlewareFixture()
        step_middleware_1 = _StepMiddlewareFixture()
        step_middleware_2 = _StepMiddlewareFixture()

        inference, step = _partition_middleware(
            [inference_middleware, step_middleware_1, step_middleware_2]
        )

        assert inference == [inference_middleware]
        assert step == [step_middleware_1, step_middleware_2]

    def test_empty(self):
        assert _partition_middleware([]) == ([], [])

    def test_rejects_unknown_middleware_type(self):
        class NotMiddleware:
            pass

        with pytest.raises(TypeError, match="InferenceMiddleware or StepMiddleware"):
            _partition_middleware([NotMiddleware()])
