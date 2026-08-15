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

    ``required_env`` is the environment variable whose presence enables the
    provider — its API key, except for Ollama where it is the server address.
    """

    id: str
    required_env: str
    default_model: str
    model_env: str

    @property
    def model(self) -> str:
        return os.environ.get(self.model_env) or self.default_model


class LiveSessionFactory(Protocol):
    def __call__(
        self, provider: Provider, **session_kwargs: Any
    ) -> AbstractAsyncContextManager[LiteLLMClient]: ...


PROVIDERS = [
    Provider(
        id="openai",
        required_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        model_env="SEFIA_E2E_OPENAI_MODEL",
    ),
    Provider(
        id="anthropic",
        required_env="ANTHROPIC_API_KEY",
        default_model="anthropic/claude-haiku-4-5",
        model_env="SEFIA_E2E_ANTHROPIC_MODEL",
    ),
    Provider(
        id="gemini",
        required_env="GEMINI_API_KEY",
        default_model="gemini/gemini-2.5-flash",
        model_env="SEFIA_E2E_GEMINI_MODEL",
    ),
    Provider(
        id="xai",
        required_env="XAI_API_KEY",
        default_model="xai/grok-3-mini",
        model_env="SEFIA_E2E_XAI_MODEL",
    ),
    Provider(
        id="mistral",
        required_env="MISTRAL_API_KEY",
        default_model="mistral/mistral-small-latest",
        model_env="SEFIA_E2E_MISTRAL_MODEL",
    ),
    Provider(
        id="groq",
        required_env="GROQ_API_KEY",
        default_model="groq/llama-3.3-70b-versatile",
        model_env="SEFIA_E2E_GROQ_MODEL",
    ),
    Provider(
        id="deepseek",
        required_env="DEEPSEEK_API_KEY",
        default_model="deepseek/deepseek-chat",
        model_env="SEFIA_E2E_DEEPSEEK_MODEL",
    ),
    Provider(
        id="ollama",
        required_env="OLLAMA_API_BASE",
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
                        not os.environ.get(provider.required_env),
                        reason=f"{provider.required_env} is not set",
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
