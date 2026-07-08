from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import glyff
import glyff_file_store
import sefia
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer
from sefia import Profile, Policy
from sefia.llm import LLMClient

from ._session_state import bind_session_state
from .policies import DefaultPolicy
from .stores import FileSessionStore, SessionStore


class SessionScope:
    """
    Manages shared configuration for Sefia sessions and provides helpers to run
    code within a configured session context.

    ``session_store_factory`` is the seam for a custom session-state
    persistence backend: it receives the session id and returns the
    :class:`SessionStore` to bind for that session. By default a
    :class:`FileSessionStore` under ``session_dir`` is used.
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        model: str | None = None,
        llm_client: LLMClient | None = None,
        policies: list[Policy] | None = None,
        profiles: list[Profile] | None = None,
        stream: bool = False,
        max_steps: int | None = 25,
        session_store_factory: Callable[[str], SessionStore] | None = None,
    ):
        self.session_dir = session_dir
        self.model = model
        self.llm_client = llm_client
        self.policies = list(policies or [])
        self.profiles = list(profiles or [])
        self.stream = stream
        self.max_steps = max_steps
        self.session_store_factory = session_store_factory

    @asynccontextmanager
    async def session(
        self,
        *,
        session_id: str,
        model: str | None = None,
        stream: bool | None = None,
        policies: list[Policy] | None = None,
        profiles: list[Profile] | None = None,
    ) -> AsyncIterator[sefia.Session]:
        """Run code within a configured Sefia session context."""
        llm_client = self.llm_client
        resolved_model = model or self.model
        resolved_stream = self.stream if stream is None else stream

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

        backend = glyff_file_store.JsonFileBackend(
            base_dir=self.session_dir / "glyff_sessions",
            session_id=session_id,
        )
        gs = glyff.Session(
            id=session_id,
            backend=backend,
            serializer=serializer,
            hasher=PydanticArgsHasher(),
        )
        if self.session_store_factory is not None:
            session_store = self.session_store_factory(session_id)
        else:
            session_store = FileSessionStore(
                base_dir=self.session_dir / "sefia_metadata" / session_id,
                serializer=serializer,
            )

        final_policies: list[Policy] = list(self.policies)
        if policies is not None:
            final_policies.extend(policies)
        final_policies.append(DefaultPolicy(max_steps=self.max_steps))

        final_profiles: list[Profile] = list(self.profiles)
        if profiles is not None:
            final_profiles.extend(profiles)

        async with gs:
            with bind_session_state(session_store):
                async with sefia.Session(
                    llm_client=llm_client,
                    glyff_session=gs,
                    policies=final_policies,
                    profiles=final_profiles,
                    stream=resolved_stream,
                ) as session:
                    yield session
