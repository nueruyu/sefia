from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import glyff
import glyff_file_store
import sefia
import sefia.stores
import sefia_litellm
from sefia.interfaces import Policy
from sefia.pydantic.glyff_serialization import SefiaArgsHasher, SefiaSerializer

from .debugging import VerbosePolicy
from .streaming import StreamingPolicy


@asynccontextmanager
async def setup_session(
    model: str,
    session_id: str,
    stream: bool,
    verbose: bool,
    session_dir: Path,
    text_block_selectors: dict[type, Callable[[Any], str]] | None = None,
) -> AsyncIterator[sefia.Session]:
    """Sets up and provides a Sefia session."""
    llm_client = sefia_litellm.LiteLLMClient(model=model)
    serializer = SefiaSerializer()

    file_client = glyff_file_store.FileClient(
        base_dir=session_dir / "glyff_sessions",
        session_id=session_id,
    )
    gs = glyff.Session(
        id=session_id,
        store=glyff_file_store.JsonFileSessionStore(
            client=file_client, serializer=serializer
        ),
        hasher=SefiaArgsHasher(),
    )
    sefia_store = sefia.stores.FileSessionStore(
        client=file_client, serializer=serializer
    )

    policies: list[Policy] = []
    if stream:
        policies.append(StreamingPolicy())
    if verbose:
        policies.append(VerbosePolicy())

    async with gs:
        async with sefia.Session(
            llm_client=llm_client,
            glyff_session=gs,
            session_store=sefia_store,
            policies=policies,
            stream=stream,
            text_block_selectors=text_block_selectors,
        ) as session:
            yield session
