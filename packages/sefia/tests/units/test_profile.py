import pytest

from sefia import Policy, Profile, profile
from sefia.testing import MockLLMClient


def test_profile_decorator_rejects_unhashable_and_none_keys() -> None:
    with pytest.raises(TypeError, match="hashable"):
        profile(["not", "hashable"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must not be None"):
        profile(None)


def test_profile_decorator_rejects_profile_as_key() -> None:
    with pytest.raises(TypeError, match="not the Profile instance itself"):
        profile(Profile(key="fast", client=MockLLMClient([])))


def test_profile_accepts_policy_sequence() -> None:
    configured = Policy()

    selected = Profile(key="profile", client=MockLLMClient([]), policies=[configured])

    assert list(selected.policies) == [configured]
