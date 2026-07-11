from collections.abc import Callable

from ..event_system import EventHandler
from .middleware import InferenceMiddleware, StepMiddleware


class Policy:
    """
    A policy that can be applied to an @infer call.

    A policy contributes two kinds of extension to an inference run:

    - Observation via ``create_handlers``, which returns event handlers that are
      notified of events but cannot steer the loop.
    - Control via ``create_middleware``, which returns middleware that wraps the
      inference run or each step and can steer the executor's loops by retrying
      or raising exceptions.

    For one-off composition, build a policy directly from factories::

        Policy(middleware=lambda: [Retrier(max_retries=5)])

    Each factory is called once per inference run, so stateful middleware
    (e.g. retry counters) is correctly scoped to a single run. Named, reusable
    policies subclass and override ``create_handlers`` / ``create_middleware``
    instead; each defaults to calling the corresponding factory, and a factory
    left unset contributes nothing.
    """

    # Class-level fallbacks so subclasses whose __init__ does not call
    # super().__init__() (e.g. dataclasses) still get empty defaults.
    _handlers_factory: Callable[[], list[EventHandler]] | None = None
    _middleware_factory: Callable[[], list[InferenceMiddleware | StepMiddleware]] | None = None

    def __init__(
        self,
        *,
        handlers: Callable[[], list[EventHandler]] | None = None,
        middleware: (
            Callable[[], list[InferenceMiddleware | StepMiddleware]] | None
        ) = None,
    ):
        self._handlers_factory = handlers
        self._middleware_factory = middleware

    def create_handlers(self) -> list[EventHandler]:
        """Create observation handlers used by this policy (default: none)."""
        return self._handlers_factory() if self._handlers_factory else []

    def create_middleware(self) -> list[InferenceMiddleware | StepMiddleware]:
        """Create control middleware used by this policy (default: none)."""
        return self._middleware_factory() if self._middleware_factory else []
