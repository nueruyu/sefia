from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import final

import glyff
import sefia
from glyff.serialization import (
    FallbackByTypeQualname,
)
from glyff_pydantic import (
    PydanticArgumentCanonicalizer,
    PydanticSerializer,
)
from sefia import HistoryStorage, Policy, Profile, ToolCollector
from sefia.llm import LLMClient

from ._session_state import bind_session_storage
from .persistence import PersistenceProvider, SQLitePersistenceProvider
from .policies import DefaultPolicy


@final
class SessionScope:
    """
    Manages shared configuration for Sefia sessions and provides helpers to run
    code within a configured session context.

    ``persistence`` creates both the glyff execution backend and Sefia's
    session-state storage so their durability semantics stay aligned. By
    default both are stored in ``.sessions/sessions.sqlite3``.

    ``history_storage`` selects where run history is persisted; defaults to the
    run's glyff metadata (:class:`~sefia.history_storages.GlyffHistoryStorage`).

    ``tool_collector`` customizes tool discovery for a run. A collector passed to
    :meth:`session` overrides the instance default; passing ``None`` there inherits
    the instance default rather than resetting it. When neither is set,
    :class:`~sefia.Session` builds its own :class:`DefaultToolCollector`.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        llm_client: LLMClient | None = None,
        policies: list[Policy] | None = None,
        profiles: list[Profile] | None = None,
        stream: bool = False,
        max_steps: int | None = 25,
        max_repair_attempts: int = 2,
        persistence: PersistenceProvider | None = None,
        history_storage: HistoryStorage | None = None,
        tool_collector: ToolCollector | None = None,
    ):
        self.model = model
        self.llm_client = llm_client
        self.policies = list(policies or [])
        self.profiles = list(profiles or [])
        self.stream = stream
        self.max_steps = max_steps
        self.max_repair_attempts = max_repair_attempts
        self.persistence = persistence or SQLitePersistenceProvider(
            ".sessions/sessions.sqlite3"
        )
        self.history_storage = history_storage
        self.tool_collector = tool_collector

    @asynccontextmanager
    async def session(
        self,
        *,
        session_id: str,
        model: str | None = None,
        stream: bool | None = None,
        policies: list[Policy] | None = None,
        profiles: list[Profile] | None = None,
        tool_collector: ToolCollector | None = None,
    ) -> AsyncGenerator[sefia.Session]:
        """Run code within a configured Sefia session context."""
        llm_client = self.llm_client
        resolved_model = model or self.model
        resolved_stream = self.stream if stream is None else stream
        resolved_tool_collector = (
            self.tool_collector if tool_collector is None else tool_collector
        )

        if llm_client is None:
            if resolved_model is None:
                raise ValueError("Either llm_client or model must be provided.")
            try:
                import sefia_litellm
            except ImportError as e:
                raise ImportError(
                    "The 'litellm' extra is required to use the default session "
                    "setup. Please install it with: pip install 'sefios[litellm]'"
                ) from e
            llm_client = sefia_litellm.LiteLLMClient(model=resolved_model)

        serializer = PydanticSerializer()

        backend = self.persistence.create_execution_backend(session_id)
        gs = glyff.Session(
            id=glyff.SessionId(session_id),
            backend=backend,
            serializer=serializer,
            argument_canonicalizer=PydanticArgumentCanonicalizer(
                FallbackByTypeQualname()
            ),
        )
        session_storage = self.persistence.create_session_storage(session_id)

        final_policies: list[Policy] = list(self.policies)
        if policies is not None:
            final_policies.extend(policies)
        final_policies.append(DefaultPolicy(max_steps=self.max_steps))

        final_profiles: list[Profile] = list(self.profiles)
        if profiles is not None:
            final_profiles.extend(profiles)

        async with gs:
            with bind_session_storage(session_storage):
                async with sefia.Session(
                    llm_client=llm_client,
                    glyff_session=gs,
                    policies=final_policies,
                    profiles=final_profiles,
                    stream=resolved_stream,
                    tool_collector=resolved_tool_collector,
                    history_storage=self.history_storage,
                    max_repair_attempts=self.max_repair_attempts,
                ) as session:
                    yield session
