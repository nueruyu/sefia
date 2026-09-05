from dataclasses import dataclass

import glyff

import sefia
from sefia import Policy, policy, profile
from sefia._authoring.metadata import KEY_PROFILE_KEY, get_metadata

infer = sefia.Domain(glyff.Domain("tests.authoring-metadata", version="1")).infer


@dataclass
class _Policy(Policy):
    count: int


def test_policy_metadata_survives_both_decorator_orders() -> None:
    @infer
    @policy(_Policy(count=3))
    async def below(value: int) -> int: ...

    @policy(_Policy(count=3))
    @infer
    async def above(value: int) -> int: ...

    for function in (below, above):
        policies = get_metadata(function)["policies"]
        assert len(policies) == 1
        assert isinstance(policies[0], _Policy)


def test_policy_preserves_metadata_added_before_infer() -> None:
    async def function(value: int) -> int: ...

    setattr(function, "__sefia_metadata__", {"other": True})
    decorated = policy(_Policy(count=2))(infer(function))

    metadata = get_metadata(decorated)
    assert metadata.get("other") is True
    assert [type(item) for item in metadata.get("policies", [])] == [_Policy]


def test_profile_metadata_survives_both_decorator_orders() -> None:
    @infer
    @profile("fast")
    async def below(value: int) -> int: ...

    @profile("smart")
    @infer
    async def above(value: int) -> int: ...

    assert get_metadata(below)[KEY_PROFILE_KEY] == "fast"
    assert get_metadata(above)[KEY_PROFILE_KEY] == "smart"
