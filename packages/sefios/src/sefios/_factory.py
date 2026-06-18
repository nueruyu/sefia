from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import glyff
import glyff_file_store
import sefia
import sefia.stores
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer
from sefia import Policy
from sefia.llm import LLMClient

from .policies._max_steps import MaxSteps


@asynccontextmanager
async def create_session(
    session_id: str,
    session_dir: Path,
    *,
    llm_client: LLMClient | None = None,
    model: str | None = None,
    stream: bool = False,
    verbose: bool = False,
    policies: list[Policy] | None = None,
    max_steps: int | None = 25,
) -> AsyncIterator[sefia.Session]:
    """Sets up and provides a Sefia session."""
    if llm_client is None:
        if model is None:
            raise ValueError("Either llm_client or model must be provided.")
        try:
            import sefia_litellm
        except ImportError as e:
            raise ImportError(
                "The 'litellm' extra is required to use the default session setup. "
                "Please install it with: pip install 'sefios[litellm]'"
            ) from e
        llm_client = sefia_litellm.LiteLLMClient(model=model)

    serializer = PydanticSerializer()

    file_client = glyff_file_store.FileClient(
        base_dir=session_dir / "glyff_sessions",
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

    final_policies: list[Policy] = list(policies) if policies is not None else []
    if max_steps is not None:
        final_policies.append(MaxSteps(count=max_steps))
    if stream:
        from .policies._streaming import StreamingPolicy

        final_policies.append(StreamingPolicy())
    if verbose:
        from .policies._debugging import VerbosePolicy

        final_policies.append(VerbosePolicy())

    async with gs:
        async with sefia.Session(
            llm_client=llm_client,
            glyff_session=gs,
            session_store=session_store,
            policies=final_policies,
            stream=stream,
        ) as session:
            yield session
