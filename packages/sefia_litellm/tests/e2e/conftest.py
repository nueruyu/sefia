"""Shared fixtures for the live-provider e2e tests.

These tests talk to real LLM providers through ``LiteLLMClient``. They are
excluded from the default ``pytest`` run by the ``e2e`` marker; run them with::

    uv run pytest packages/sefia_litellm -m e2e

Each provider runs only when its enabling environment variable is set (an API
key, or the server address for Ollama), and its model can be overridden via
``SEFIA_E2E_<PROVIDER>_MODEL``.
"""

import os
import uuid
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import pytest
from sefia.testing import memory_session

from sefia_litellm import LiteLLMClient


@dataclass(frozen=True)
class Provider:
    """One live provider the e2e suite can run against.

    ``required_envs`` lists the environment variables whose presence enables
    the provider — API keys, except for Ollama where it is the server address.
    """

    id: str
    required_envs: tuple[str, ...]
    default_model: str
    model_env: str

    @property
    def model(self) -> str:
        return os.environ.get(self.model_env) or self.default_model

    @property
    def enabled(self) -> bool:
        return any(os.environ.get(name) for name in self.required_envs)


class LiveSessionFactory(Protocol):
    def __call__(
        self, provider: Provider, **session_kwargs: Any
    ) -> AbstractAsyncContextManager[LiteLLMClient]: ...


PROVIDERS = [
    Provider(
        id="openai",
        required_envs=("OPENAI_API_KEY",),
        default_model="gpt-4o-mini",
        model_env="SEFIA_E2E_OPENAI_MODEL",
    ),
    Provider(
        id="anthropic",
        required_envs=("ANTHROPIC_API_KEY",),
        default_model="anthropic/claude-haiku-4-5",
        model_env="SEFIA_E2E_ANTHROPIC_MODEL",
    ),
    Provider(
        id="gemini",
        required_envs=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        default_model="gemini/gemini-2.5-flash",
        model_env="SEFIA_E2E_GEMINI_MODEL",
    ),
    Provider(
        id="xai",
        required_envs=("XAI_API_KEY",),
        default_model="xai/grok-3-mini",
        model_env="SEFIA_E2E_XAI_MODEL",
    ),
    Provider(
        id="mistral",
        required_envs=("MISTRAL_API_KEY",),
        default_model="mistral/mistral-small-latest",
        model_env="SEFIA_E2E_MISTRAL_MODEL",
    ),
    Provider(
        id="groq",
        required_envs=("GROQ_API_KEY",),
        default_model="groq/llama-3.3-70b-versatile",
        model_env="SEFIA_E2E_GROQ_MODEL",
    ),
    Provider(
        id="deepseek",
        required_envs=("DEEPSEEK_API_KEY",),
        default_model="deepseek/deepseek-chat",
        model_env="SEFIA_E2E_DEEPSEEK_MODEL",
    ),
    Provider(
        id="ollama",
        required_envs=("OLLAMA_API_BASE",),
        default_model="ollama/llama3.1",
        model_env="SEFIA_E2E_OLLAMA_MODEL",
    ),
]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrizes every test taking a ``provider`` argument over PROVIDERS,
    skipping providers whose enabling environment variable is absent."""
    if "provider" in metafunc.fixturenames:
        metafunc.parametrize(
            "provider",
            [
                pytest.param(
                    provider,
                    id=provider.id,
                    marks=pytest.mark.skipif(
                        not provider.enabled,
                        reason=f"{' or '.join(provider.required_envs)} is not set",
                    ),
                )
                for provider in PROVIDERS
            ],
        )


@pytest.fixture
def live_session() -> LiveSessionFactory:
    """Async-context factory: a sefia Session backed by the provider's real API.

    A unique session id per use, so nothing is replayed between tests.
    """

    @asynccontextmanager
    async def factory(provider: Provider, **session_kwargs: Any):
        client = LiteLLMClient(model=provider.model)
        async with memory_session(
            client, session_id=f"e2e-{provider.id}-{uuid.uuid4()}", **session_kwargs
        ):
            yield client

    return factory
