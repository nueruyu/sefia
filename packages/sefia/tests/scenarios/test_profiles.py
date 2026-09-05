import sefia
from dataclasses import dataclass
from enum import Enum, auto

import glyff
import pytest

from sefia import Domain, Policy, Profile, policy, profile
from sefia.event_system import Event, EventHandler
from sefia.llm import LLMCompletion
from sefia.testing import MockLLMClient, memory_session, result_completion

infer = sefia.Domain(
    glyff.Domain("packages.sefia.tests.scenarios.test_profiles", version="1")
).infer


@dataclass
class Report:
    topic: str
    summary: str
    sources: list[str]


def _result(summary: str) -> LLMCompletion:
    return result_completion(Report(topic="t", summary=summary, sources=[]))


@dataclass
class _LabelPolicy(Policy):
    """A policy that appends its label to a shared log when its handlers build."""

    label: str
    log: list[str]

    def create_handlers(self) -> list[EventHandler[Event]]:
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


async def test_infer_uses_selected_profile_client():
    """An @infer decorated with @profile runs on the profile's client, while an
    undecorated @infer on the same agent uses the session default."""

    default_llm = MockLLMClient(completions=[_result("from default")])
    fast_llm = MockLLMClient(completions=[_result("from fast")])

    async with memory_session(
        default_llm,
        session_id="profiles",
        profiles=[Profile(key="fast", client=fast_llm)],
    ):
        agent = _ProfileAgent()
        default_report = await agent.with_default(topic="t")
        fast_report = await agent.with_fast(topic="t")

    # Each call landed on the matching client, not the other.
    assert default_report.summary == "from default"
    assert fast_report.summary == "from fast"
    assert len(default_llm.requests) == 1
    assert len(fast_llm.requests) == 1


async def test_policy_layering_session_profile_function():
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

    fast_llm = MockLLMClient(completions=[_result("ok")])
    async with memory_session(
        fast_llm,
        session_id="layering",
        policies=[_LabelPolicy(label="session", log=log)],
        profiles=[
            Profile(
                key="fast",
                client=fast_llm,
                policies=[_LabelPolicy(label="profile", log=log)],
            )
        ],
    ):
        await Agent().step(topic="t")

    assert log == ["session", "profile", "function"]


async def test_domain_default_profile_and_function_override():
    fast_llm = MockLLMClient(completions=[_result("from domain")])
    smart_llm = MockLLMClient(completions=[_result("from function")])
    reports = Domain(glyff.Domain("tests.reports", version="1"), default_profile="fast")

    @reports.infer(
        name="test_domain_default_profile_and_function_override.with_domain_default"
    )
    async def with_domain_default(topic: str) -> Report: ...

    @reports.infer(
        name="test_domain_default_profile_and_function_override.with_function_override"
    )
    @profile("smart")
    async def with_function_override(topic: str) -> Report: ...

    async with memory_session(
        MockLLMClient(completions=[]),
        session_id="domain-profiles",
        profiles=[
            Profile(key="fast", client=fast_llm),
            Profile(key="smart", client=smart_llm),
        ],
    ):
        domain_report = await with_domain_default(topic="t")
        function_report = await with_function_override(topic="t")

    assert domain_report.summary == "from domain"
    assert function_report.summary == "from function"


async def test_policy_layering_includes_domain_defaults():
    log: list[str] = []
    reports = Domain(
        glyff.Domain("tests.reports", version="1"),
        default_profile="fast",
        policies=[_LabelPolicy(label="domain", log=log)],
    )

    @reports.infer
    @policy(_LabelPolicy(label="function", log=log))
    async def step(topic: str) -> Report: ...

    llm = MockLLMClient(completions=[_result("ok")])
    async with memory_session(
        llm,
        session_id="domain-policy-layering",
        policies=[_LabelPolicy(label="session", log=log)],
        profiles=[
            Profile(
                key="fast",
                client=llm,
                policies=[_LabelPolicy(label="profile", log=log)],
            )
        ],
    ):
        await step(topic="t")

    assert log == ["session", "domain", "profile", "function"]


async def test_unknown_profile_raises():
    """Referencing a profile the session does not register fails fast at call
    time with the list of registered profiles."""

    default_llm = MockLLMClient(completions=[_result("unused")])
    async with memory_session(
        default_llm,
        session_id="unknown",
        profiles=[Profile(key="fast", client=default_llm)],
    ):
        with pytest.raises(RuntimeError, match="Unknown profile 'missing'"):
            await _MissingProfileAgent().step(topic="t")


class _Models(Enum):
    FAST = auto()
    SMART = auto()


async def test_enum_key_selects_profile():
    """A profile key can be any hashable, e.g. an Enum member."""

    class Agent:
        @infer
        @profile(_Models.SMART)
        async def step(self, topic: str) -> Report:
            """Runs on the profile keyed by an enum member."""
            ...

    default_llm = MockLLMClient(completions=[_result("default")])
    smart_llm = MockLLMClient(completions=[_result("from smart")])

    async with memory_session(
        default_llm,
        session_id="enum-key",
        profiles=[Profile(key=_Models.SMART, client=smart_llm)],
    ):
        report = await Agent().step(topic="t")

    assert report.summary == "from smart"
    assert len(smart_llm.requests) == 1
    assert len(default_llm.requests) == 0


async def test_unknown_enum_key_lists_registered():
    """An unknown key reports the registered profiles by repr (no sorting, since
    arbitrary keys are not necessarily orderable)."""

    class Agent:
        @infer
        @profile(_Models.FAST)
        async def step(self, topic: str) -> Report:
            """Selects a profile keyed by an enum member that is not registered."""
            ...

    default_llm = MockLLMClient(completions=[_result("unused")])
    async with memory_session(
        default_llm,
        session_id="enum-miss",
        profiles=[Profile(key=_Models.SMART, client=default_llm)],
    ):
        with pytest.raises(RuntimeError, match=r"Unknown profile <_Models.FAST"):
            await Agent().step(topic="t")
