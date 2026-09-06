import glyff
import pytest
from glyff import ArgumentCanonicalizer, Serializer
from glyff.store import MemoryBackend

from sefia import Profile, Session
from sefia.testing import MockLLMClient


def test_duplicate_profile_keys_are_rejected(
    serializer: Serializer, hasher: ArgumentCanonicalizer
) -> None:
    first = MockLLMClient([])
    second = MockLLMClient([])

    with pytest.raises(ValueError, match="Duplicate profile key: 'duplicate'"):
        Session(
            llm_client=first,
            glyff_session=glyff.Session(
                id=glyff.SessionId("duplicate-profile"),
                backend=MemoryBackend(),
                serializer=serializer,
                argument_canonicalizer=hasher,
            ),
            profiles=[
                Profile(key="duplicate", client=first),
                Profile(key="duplicate", client=second),
            ],
        )
