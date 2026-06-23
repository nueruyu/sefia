import json
from dataclasses import dataclass

import glyff
import pytest
from glyff import ArgsHasher, Serializer
from glyff.store import MemoryClient
from glyff.store import MemorySessionStore as GlyffMemoryStore

from sefia import ModelProfile, Policy, Session, infer, policy, profile
from sefia._metadata import PROFILE_KEY, get_metadata
from sefia.event_system import EventHandler
from sefia.llm import LLMResponse
from sefia.stores import MemorySessionStore as SefiaMemoryStore

from ..conftest import MockLLMClient, Report


def _make_stores(serializer):
    client = MemoryClient()
    return (
        GlyffMemoryStore(client=client, serializer=serializer),
        SefiaMemoryStore(client=client, serializer=serializer),
    )


def _final_answer(summary: str) -> LLMResponse:
    return LLMResponse(
        content=json.dumps(
            {"final_answer": {"topic": "t", "summary": summary, "sources": []}}
        )
    )


@dataclass
class _LabelPolicy(Policy):
    """A policy that appends its label to a shared log when its handlers build."""

    label: str
    log: list[str]

    def create_handlers(self) -> list[EventHandler]:
        self.log.append(self.label)
        return []


class _ProfileAgent:
    @infer
    async def with_default(self, topic: str) -> Report:
        """Run on the session default model."""
        ...

    @infer
    @profile("fast")
    async def with_fast(self, topic: str) -> Report:
        """Run on the 'fast' profile."""
        ...


class _MissingProfileAgent:
    @infer
    @profile("missing")
    async def step(self, topic: str) -> Report:
        """Selects a profile that was never registered."""
        ...


def test_profile_attaches_name_metadata():
    """`@profile` records the profile name under the metadata "profile" key, no
    matter where it sits relative to @infer."""

    @infer
    @profile("fast")
    async def below(value: int) -> int:
        """Profile selected below @infer."""
        ...

    @profile("smart")
    @infer
    async def above(value: int) -> int:
        """Profile selected above @infer."""
        ...

    assert get_metadata(below)[PROFILE_KEY] == "fast"
    assert get_metadata(above)[PROFILE_KEY] == "smart"


def test_profile_rejects_non_string():
    """@profile raises a clear error when given something other than a name."""
    with pytest.raises(TypeError):
        profile(object())  # type: ignore


def test_model_profile_normalizes_policies_to_tuple():
    """ModelProfile accepts a list of policies but stores an immutable tuple."""
    p = _LabelPolicy(label="x", log=[])
    prof = ModelProfile(name="p", client=MockLLMClient(responses=[]), policies=[p])
    assert prof.policies == (p,)


async def test_infer_uses_selected_profile_client(
    serializer: Serializer, hasher: ArgsHasher
):
    """An @infer decorated with @profile runs on the profile's client, while an
    undecorated @infer on the same agent uses the session default."""

    default_llm = MockLLMClient(responses=[_final_answer("from default")])
    fast_llm = MockLLMClient(responses=[_final_answer("from fast")])

    glyff_store, sefia_store = _make_stores(serializer)
    async with glyff.Session(id="profiles", store=glyff_store, hasher=hasher) as gs:
        async with Session(
            llm_client=default_llm,
            glyff_session=gs,
            session_store=sefia_store,
            profiles=[ModelProfile(name="fast", client=fast_llm)],
        ):
            agent = _ProfileAgent()
            default_report = await agent.with_default(topic="t")
            fast_report = await agent.with_fast(topic="t")

    # Each call landed on the matching client, not the other.
    assert default_report.summary == "from default"
    assert fast_report.summary == "from fast"
    assert len(default_llm.requests) == 1
    assert len(fast_llm.requests) == 1


async def test_policy_layering_session_profile_function(
    serializer: Serializer, hasher: ArgsHasher
):
    """Policies are collected most-general first: session -> profile -> function,
    so the function's own @policy sits closest to the call."""

    log: list[str] = []

    class Agent:
        @infer
        @policy(_LabelPolicy(label="function", log=log))
        @profile("fast")
        async def step(self, topic: str) -> Report:
            """Selects a profile and adds its own policy."""
            ...

    fast_llm = MockLLMClient(responses=[_final_answer("ok")])
    glyff_store, sefia_store = _make_stores(serializer)
    async with glyff.Session(id="layering", store=glyff_store, hasher=hasher) as gs:
        async with Session(
            llm_client=fast_llm,
            glyff_session=gs,
            session_store=sefia_store,
            policies=[_LabelPolicy(label="session", log=log)],
            profiles=[
                ModelProfile(
                    name="fast",
                    client=fast_llm,
                    policies=[_LabelPolicy(label="profile", log=log)],
                )
            ],
        ):
            await Agent().step(topic="t")

    assert log == ["session", "profile", "function"]


async def test_unknown_profile_raises(serializer: Serializer, hasher: ArgsHasher):
    """Referencing a profile the session does not register fails fast at call
    time with the list of registered profiles."""

    default_llm = MockLLMClient(responses=[_final_answer("unused")])
    glyff_store, sefia_store = _make_stores(serializer)
    async with glyff.Session(id="unknown", store=glyff_store, hasher=hasher) as gs:
        async with Session(
            llm_client=default_llm,
            glyff_session=gs,
            session_store=sefia_store,
            profiles=[ModelProfile(name="fast", client=default_llm)],
        ):
            with pytest.raises(RuntimeError, match="Unknown profile 'missing'"):
                await _MissingProfileAgent().step(topic="t")


def test_duplicate_profile_names_rejected(
    serializer: Serializer, hasher: ArgsHasher
):
    """The Session rejects two profiles sharing a name up front."""
    glyff_store, sefia_store = _make_stores(serializer)
    a = MockLLMClient(responses=[])
    b = MockLLMClient(responses=[])
    with pytest.raises(ValueError, match="Duplicate profile name: 'dup'"):
        Session(
            llm_client=a,
            glyff_session=glyff.Session(id="dup", store=glyff_store, hasher=hasher),
            session_store=sefia_store,
            profiles=[
                ModelProfile(name="dup", client=a),
                ModelProfile(name="dup", client=b),
            ],
        )
