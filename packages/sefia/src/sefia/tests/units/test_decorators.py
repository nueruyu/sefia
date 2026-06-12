import pytest

from sefia._decorators import _partition_middleware
from sefia.middleware.max_steps import StepLimiter
from sefia.middleware.retry import Retrier
from sefia.middleware.stagnation import StagnationDetector


class TestPartitionMiddleware:
    def test_splits_by_seam(self):
        retrier = Retrier(max_retries=1)
        limiter = StepLimiter(max_steps=5)
        detector = StagnationDetector(max_repeats=2)

        inference, step = _partition_middleware([retrier, limiter, detector])

        assert inference == [retrier]
        assert step == [limiter, detector]

    def test_empty(self):
        assert _partition_middleware([]) == ([], [])

    def test_rejects_unknown_middleware_type(self):
        class NotMiddleware:
            pass

        with pytest.raises(TypeError, match="InferenceMiddleware or StepMiddleware"):
            _partition_middleware([NotMiddleware()])
