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
    ):
        self.session_dir = session_dir
        self.default_model = default_model
        self.llm_client = llm_client
        self.default_policies = list(default_policies or [])

    def __call__(
        self, func: Callable[..., Coroutine[Any, Any, Any]]
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        """
        Decorator that wraps an async function to run within a sefia session.
        It pulls session-related parameters from the decorated function's
        keyword arguments.
        """

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            session_id = kwargs.get("session_id")
            if not session_id or not isinstance(session_id, str):
                raise TypeError(
                    "The decorated function must have a 'session_id: str' keyword argument."
                )

            model = kwargs.get("model", self.default_model)
            stream = kwargs.get("stream", False)
            verbose = kwargs.get("verbose", False)
            policies = list(self.default_policies)

            async with create_session(
                session_id=session_id,
                session_dir=self.session_dir,
                llm_client=self.llm_client,
                model=model,
                stream=stream,
                verbose=verbose,
                policies=policies,
            ):
                return await func(*args, **kwargs)

        return wrapper
