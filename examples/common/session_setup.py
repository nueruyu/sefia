from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import glyff
import glyff_file_store
import sefia
import sefia.stores
import sefia_litellm
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer
from sefia import Policy
from sefia.policies import MaxSteps

from .debugging import VerbosePolicy
from .streaming import StreamingPolicy


@asynccontextmanager
async def setup_session(
    model: str,
    session_id: str,
    stream: bool,
    verbose: bool,
    session_dir: Path,
) -> AsyncIterator[sefia.Session]:
    """Sets up and provides a Sefia session."""
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

    policies: list[Policy] = [MaxSteps(count=25)]
    if stream:
        policies.append(StreamingPolicy())
    if verbose:
        policies.append(VerbosePolicy())

    async with gs:
        async with sefia.Session(
            llm_client=llm_client,
            glyff_session=gs,
            session_store=session_store,
            policies=policies,
            stream=stream,
        ) as session:
            yield session
