from collections.abc import Callable

from sefia._interfaces import InferenceMiddleware, Policy, StepMiddleware
from sefia.event_system import EventHandler


class CustomPolicy(Policy):
    """
    A policy that accepts handler and middleware factories for customization,
    removing the need to subclass Policy for each new handler or middleware.

    Each factory is called once per inference run, so stateful middleware
    (e.g. retry counters) is correctly scoped to a single run.

    Example::

        @policy(CustomPolicy(middleware=lambda: [Retrier(max_retries=5)]))
        @infer
        async def step(...): ...
    """

    def __init__(
        self,
        *,
        handlers: Callable[[], list[EventHandler]] | None = None,
        middleware: (
            Callable[[], list[InferenceMiddleware | StepMiddleware]] | None
        ) = None,
    ):
        self._handlers = handlers
        self._middleware = middleware

    def create_handlers(self) -> list[EventHandler]:
        return self._handlers() if self._handlers else []

    def create_middleware(self) -> list[InferenceMiddleware | StepMiddleware]:
        return self._middleware() if self._middleware else []
