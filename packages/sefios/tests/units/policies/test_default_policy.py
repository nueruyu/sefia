from sefios.middleware import (
    InputCallComposer,
    StagnationDetector,
    StepLimiter,
)
from sefios.policies import DefaultPolicy


def test_default_policy_includes_input_composition():
    middleware = DefaultPolicy().create_middleware()

    assert [type(m) for m in middleware] == [
        StepLimiter,
        StagnationDetector,
        InputCallComposer,
    ]
