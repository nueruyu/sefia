from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import glyff
import glyff_file_store
import sefia
import sefia.stores
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer
from sefia import Policy
from sefia.llm import LLMClient

from .policies import MaxSteps, StreamingPolicy, VerbosePolicy

# Sentinel distinguishing "not overridden" from an explicit ``max_steps=None``
# (which is a meaningful value meaning "no step limit").
_UNSET: Any = object()


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
        stream: bool = False,
        verbose: bool = False,
        max_steps: int | None = 25,
    ):
        self.session_dir = session_dir
        self.model = model
        self.llm_client = llm_client
        self.policies = list(policies or [])
        self.stream = stream
        self.verbose = verbose
        self.max_steps = max_steps

    @asynccontextmanager
    async def session(
        self,
        *,
        session_id: str,
        model: str | None = None,
        stream: bool | None = None,
        verbose: bool | None = None,
        max_steps: int | None | Any = _UNSET,
    ) -> AsyncIterator[sefia.Session]:
        """Run code within a configured Sefia session context."""
        llm_client = self.llm_client
        resolved_model = model or self.model
        resolved_stream = self.stream if stream is None else stream
        resolved_verbose = self.verbose if verbose is None else verbose
        resolved_max_steps = self.max_steps if max_steps is _UNSET else max_steps

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
        if resolved_max_steps is not None:
            final_policies.append(MaxSteps(count=resolved_max_steps))
        if resolved_stream:
            final_policies.append(StreamingPolicy())
        if resolved_verbose:
            final_policies.append(VerbosePolicy())

        async with gs:
            async with sefia.Session(
                llm_client=llm_client,
                glyff_session=gs,
                session_store=session_store,
                policies=final_policies,
                stream=resolved_stream,
            ) as session:
                yield session
