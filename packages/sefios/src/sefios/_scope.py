from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import glyff
import glyff_file_store
import sefia
import sefia.stores
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer
from sefia import ModelProfile, Policy
from sefia.llm import LLMClient

from .policies import DefaultPolicy


class SessionScope:
    """
    Manages shared configuration for Sefia sessions and provides helpers to run
    code within a configured session context.
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        model: str | None = None,
        llm_client: LLMClient | None = None,
        policies: list[Policy] | None = None,
        profiles: list[ModelProfile] | None = None,
        stream: bool = False,
        max_steps: int | None = 25,
    ):
        self.session_dir = session_dir
        self.model = model
        self.llm_client = llm_client
        self.policies = list(policies or [])
        self.profiles = list(profiles or [])
        self.stream = stream
        self.max_steps = max_steps

    @asynccontextmanager
    async def session(
        self,
        *,
        session_id: str,
        model: str | None = None,
        stream: bool | None = None,
        policies: list[Policy] | None = None,
        profiles: list[ModelProfile] | None = None,
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

        file_client = glyff_file_store.FileClient(
            base_dir=self.session_dir / "glyff_sessions",
            session_id=session_id,
        )
        gs = glyff.Session(
            id=session_id,
            store=glyff_file_store.JsonFileSessionStore(
                client=file_client, serializer=serializer
            ),
            hasher=PydanticArgsHasher(),
        )
        session_store = sefia.stores.FileSessionStore(
            client=file_client, serializer=serializer
        )

        final_policies: list[Policy] = list(self.policies)
        if policies is not None:
            final_policies.extend(policies)
        final_policies.append(DefaultPolicy(max_steps=self.max_steps))

        final_profiles: list[ModelProfile] = list(self.profiles)
        if profiles is not None:
            final_profiles.extend(profiles)

        async with gs:
            async with sefia.Session(
                llm_client=llm_client,
                glyff_session=gs,
                session_store=session_store,
                policies=final_policies,
                profiles=final_profiles,
                stream=resolved_stream,
            ) as session:
                yield session
