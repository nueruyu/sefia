from sefios.middleware import (
    HumanInputCallComposer,
    StagnationDetector,
    StepLimiter,
)
from sefios.policies import DefaultPolicy


def test_default_policy_includes_human_input_composition():
    middleware = DefaultPolicy().create_middleware()

    assert [type(m) for m in middleware] == [
        StepLimiter,
        StagnationDetector,
        HumanInputCallComposer,
    ]
