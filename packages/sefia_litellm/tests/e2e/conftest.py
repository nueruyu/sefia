"""Shared fixtures for the live-provider e2e tests.

These tests talk to real LLM providers through ``LiteLLMClient``. They are
excluded from the default ``pytest`` run by the ``e2e`` marker (see the
``-m "not e2e"`` addopts); run them explicitly with::

    uv run pytest packages/sefia_litellm -m e2e

Each provider is skipped unless its API key is present in the environment, so
``-m e2e`` runs whatever subset of providers the environment is configured for.
The model per provider can be overridden with the ``SEFIA_E2E_<PROVIDER>_MODEL``
environment variables.
"""

import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

import glyff
import pytest
from glyff.store import MemoryBackend
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer
from sefia import Session

from sefia_litellm import LiteLLMClient


@dataclass(frozen=True)
class Provider:
    """One live provider the e2e suite can run against."""

    id: str
    api_key_env: str
    default_model: str
    model_env: str

    @property
    def model(self) -> str:
        return os.environ.get(self.model_env) or self.default_model


PROVIDERS = [
    Provider(
        id="openai",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        model_env="SEFIA_E2E_OPENAI_MODEL",
    ),
    Provider(
        id="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="anthropic/claude-opus-4-8",
        model_env="SEFIA_E2E_ANTHROPIC_MODEL",
    ),
    Provider(
        id="gemini",
        api_key_env="GEMINI_API_KEY",
        default_model="gemini/gemini-2.5-flash",
        model_env="SEFIA_E2E_GEMINI_MODEL",
    ),
]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrizes every test taking a ``provider`` argument over PROVIDERS.

    Providers whose API key is absent from the environment are skipped, so the
    suite runs against whatever subset the environment is configured for. (This
    tree is not a package, so parametrization lives here rather than in an
    importable helper.)
    """
    if "provider" in metafunc.fixturenames:
        metafunc.parametrize(
            "provider",
            [
                pytest.param(
                    provider,
                    id=provider.id,
                    marks=pytest.mark.skipif(
                        not os.environ.get(provider.api_key_env),
                        reason=f"{provider.api_key_env} is not set",
                    ),
                )
                for provider in PROVIDERS
            ],
        )


@pytest.fixture
def live_session():
    """Async-context factory: a sefia Session backed by the provider's real API.

    In-memory glyff backend with a unique session id per use, so nothing is
    replayed between tests and nothing touches disk.
    """

    @asynccontextmanager
    async def factory(provider: Provider, **session_kwargs):
        client = LiteLLMClient(model=provider.model)
        async with glyff.Session(
            id=f"e2e-{provider.id}-{uuid.uuid4()}",
            backend=MemoryBackend(),
            serializer=PydanticSerializer(),
            hasher=PydanticArgsHasher(),
        ) as glyff_session:
            async with Session(
                llm_client=client, glyff_session=glyff_session, **session_kwargs
            ):
                yield client

    return factory
