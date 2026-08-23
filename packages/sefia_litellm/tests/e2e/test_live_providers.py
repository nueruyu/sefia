"""Live end-to-end tests: the full sefia stack against real LLM providers.

Each test drives ``@infer`` through ``LiteLLMClient`` and a real provider API,
covering the three decision shapes the framework produces: a plain string
result, a structured (dataclass) result, and a tool-call round trip. Assertions
are deliberately loose about wording — models are nondeterministic — and strict
about the mechanics (types, tool dispatch, sentinel values).

See ``conftest.py`` for how providers are selected and skipped.
"""

import glyff
import sefia

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

import pytest
from sefia import Tools
from sefia.llm import LLMDecisionMode

from sefia_litellm import LiteLLMClient


class Provider(Protocol):
    id: str

    @property
    def model(self) -> str: ...


class LiveSessionFactory(Protocol):
    def __call__(
        self, provider: Provider, **session_kwargs: Any
    ) -> AbstractAsyncContextManager[LiteLLMClient]: ...


infer = sefia.Domain(
    glyff.Domain("packages.sefia_litellm.tests.e2e.test_live_providers", version="1")
).infer

pytestmark = pytest.mark.e2e

# An arbitrary value no model can guess: if it shows up in the final answer,
# it can only have come through the tool-call round trip.
_SENTINEL = "XK-7391-QZ"


@infer
async def echo_word(word: str) -> str:
    """Reply with exactly the given word, lowercase, and nothing else."""
    ...


@dataclass
class Capital:
    city: str
    country: str


@infer
async def capital_of(country: str) -> Capital:
    """Return the capital city of the given country."""
    ...


@dataclass
class VaultToolkit:
    """Holds values retrievable only by key."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def read_vault(self, key: str) -> str:
        """Read the stored value for the given vault key."""
        self.calls.append(key)
        return _SENTINEL if key == "launch-code" else "unknown key"


class VaultAgent:
    _vault: Tools[VaultToolkit]

    def __init__(self, vault: VaultToolkit) -> None:
        self._vault = vault

    @infer
    async def fetch_launch_code(self) -> str:
        """Read the vault value stored under the key 'launch-code' using the
        available tool, then answer with that exact value and nothing else."""
        ...


async def test_plain_string_result(
    provider: Provider, live_session: LiveSessionFactory
) -> None:
    async with live_session(provider):
        answer = await echo_word(word="pong")

    assert isinstance(answer, str)
    assert "pong" in answer.lower()


async def test_structured_output(
    provider: Provider, live_session: LiveSessionFactory
) -> None:
    async with live_session(provider):
        capital = await capital_of(country="Japan")

    assert isinstance(capital, Capital)
    assert "tokyo" in capital.city.lower()
    assert capital.country.strip()


async def test_tool_call_round_trip(
    provider: Provider, live_session: LiveSessionFactory
) -> None:
    vault = VaultToolkit()

    async with live_session(provider):
        answer = await VaultAgent(vault).fetch_launch_code()

    # The tool was actually dispatched with the decoded arguments...
    assert "launch-code" in vault.calls
    # ...and its result flowed back through the history into the final answer.
    assert _SENTINEL in answer


async def test_native_tool_call_round_trip(
    provider: Provider, live_session: LiveSessionFactory
) -> None:
    vault = VaultToolkit()

    async with live_session(provider, decision_mode=LLMDecisionMode.NATIVE_TOOLS):
        answer = await VaultAgent(vault).fetch_launch_code()

    assert "launch-code" in vault.calls
    assert _SENTINEL in answer
