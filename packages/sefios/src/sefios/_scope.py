import functools
import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Coroutine

from sefia import Policy, Session
from sefia.llm import LLMClient

from ._factory import create_session

# Sentinel distinguishing "not overridden" from an explicit ``max_steps=None``
# (which is a meaningful value meaning "no step limit").
_UNSET: Any = object()

# Keyword arguments the session decorator consumes before delegating to the
# wrapped function. A decorated function may not declare parameters of these
# names, otherwise its arguments would be silently swallowed here.
_RESERVED_SESSION_KWARGS = ("session_id", "model", "stream", "verbose", "max_steps")


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

    def __call__(
        self, func: Callable[..., Coroutine[Any, Any, Any]]
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        """
        Decorator that wraps an async function to run within a sefia session.
        It consumes session-related keyword arguments before calling the wrapped
        function.
        """
        collisions = [
            name
            for name in _RESERVED_SESSION_KWARGS
            if name in inspect.signature(func).parameters
        ]
        if collisions:
            raise TypeError(
                f"Cannot decorate {func.__qualname__!r} with SefiaScope: its "
                f"parameters {collisions} collide with reserved session keyword "
                "arguments and would be swallowed by the session wrapper."
            )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            session_id = kwargs.pop("session_id", None)
            if not session_id or not isinstance(session_id, str):
                raise TypeError(
                    "The decorated function must be called with a 'session_id: str' keyword argument."
                )

            model = kwargs.pop("model", None)
            stream = kwargs.pop("stream", None)
            verbose = kwargs.pop("verbose", None)
            max_steps = kwargs.pop("max_steps", _UNSET)

            async with self.session(
                session_id=session_id,
                model=model,
                stream=stream,
                verbose=verbose,
                max_steps=max_steps,
            ):
                return await func(*args, **kwargs)

        return wrapper
