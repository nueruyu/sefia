import functools
from pathlib import Path
from typing import Any, Callable, Coroutine

from sefia import Policy
from sefia.llm import LLMClient

from ._factory import create_session


class SefiaScope:
    """
    Manages shared configuration for Sefia sessions and provides a decorator
    to run functions within a configured session context.
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        default_model: str | None = None,
        llm_client: LLMClient | None = None,
        default_policies: list[Policy] | None = None,
        default_stream: bool = False,
        default_verbose: bool = False,
        max_steps: int | None = 25,
    ):
        self.session_dir = session_dir
        self.default_model = default_model
        self.llm_client = llm_client
        self.default_policies = list(default_policies or [])
        self.default_stream = default_stream
        self.default_verbose = default_verbose
        self.max_steps = max_steps

    def __call__(
        self, func: Callable[..., Coroutine[Any, Any, Any]]
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        """
        Decorator that wraps an async function to run within a sefia session.
        It consumes session-related keyword arguments before calling the wrapped
        function.
        """

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            session_id = kwargs.pop("session_id", None)
            if not session_id or not isinstance(session_id, str):
                raise TypeError(
                    "The decorated function must be called with a 'session_id: str' keyword argument."
                )

            model = kwargs.pop("model", self.default_model)
            stream = kwargs.pop("stream", self.default_stream)
            verbose = kwargs.pop("verbose", self.default_verbose)
            policies = list(self.default_policies)

            async with create_session(
                session_id=session_id,
                session_dir=self.session_dir,
                llm_client=self.llm_client,
                model=model,
                stream=stream,
                verbose=verbose,
                policies=policies,
                max_steps=self.max_steps,
            ):
                return await func(*args, **kwargs)

        return wrapper
