from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sefia import Policy, Session
from sefia.llm import LLMClient

from ._factory import create_session

# Sentinel distinguishing "not overridden" from an explicit ``max_steps=None``
# (which is a meaningful value meaning "no step limit").
_UNSET: Any = object()


class SefiaScope:
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
    ) -> AsyncIterator[Session]:
        """Run code within a configured Sefia session context."""
        async with create_session(
            session_id=session_id,
            session_dir=self.session_dir,
            llm_client=self.llm_client,
            model=model or self.model,
            stream=self.stream if stream is None else stream,
            verbose=self.verbose if verbose is None else verbose,
            policies=list(self.policies),
            max_steps=self.max_steps if max_steps is _UNSET else max_steps,
        ) as session:
            yield session
