from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

import typer
from sefia import Policy
from sefia.exceptions import InferenceError, PauseException
from sefia_typer import CLIReporter
from sefia_typer.exceptions import UnknownSessionError as CLIUnknownSessionError
from typing_extensions import final

from .._interaction_context import get_interaction_channel
from .._scope import SessionScope
from ..exceptions import InteractionRequired
from ..handlers import CostCalculator
from ..interactions import (
    InteractionChannel,
    InteractionRequest,
    JsonValue,
)
from ..persistence import MemoryPersistence, PersistenceProvider
from ..sessions import (
    ActiveSessionStore,
    MemoryActiveSessionStore,
    SessionManager,
    UnknownSessionError,
)
from ..tools import Input, Output
from ._cost_reporter import CostReportingCLIReporter
from ._reporting import CLIReporting

_USE_DEFAULT_REPORTER = object()


@final
class SefiaCLISession:
    """Operations available inside a Sefia CLI session context."""

    def __init__(self, *, channel: InteractionChannel):
        self._channel = channel

    async def resolve_interaction(self, interaction_id: str, result: JsonValue) -> None:
        """Resolve an existing interaction with an immutable JSON result."""
        await self._channel.resolve(interaction_id, result)

    async def pending_interactions(self) -> list[InteractionRequest]:
        return await self._channel.pending()


@final
class SefiaCLI:
    """Creates Sefia session contexts for Typer commands.

    The integration facade over the ``sefia_typer`` building blocks: it wires
    generic interaction reporting and :class:`Input` previews to session
    storage, runs sessions through a :class:`SessionScope` (with cost
    accounting installed), and maps pauses and inference errors to CLI exit
    codes.
    """

    def __init__(
        self,
        *,
        reporter: CLIReporter | None | object = _USE_DEFAULT_REPORTER,
        model: str | None = None,
        stream: bool = True,
        max_steps: int | None = 25,
        policies: list[Policy] | None = None,
        persistence: PersistenceProvider | None = None,
        active_session_store: ActiveSessionStore | None = None,
    ):
        persistence = persistence or MemoryPersistence()
        active_session_store = active_session_store or MemoryActiveSessionStore()
        self._reporter = self._resolve_reporter(reporter)
        self._reporting = CLIReporting(self._reporter)
        self._session_manager = SessionManager(
            persistence.create_session_registry(),
            active_session_store,
        )
        self._input_tool = Input(
            on_prompt_delta=self._reporting.input_prompt_delta,
        )
        self._output_tool = Output(
            on_output=self._reporting.output,
            on_message_delta=self._reporting.output_message_delta,
        )

        scope_policies: list[Policy] = [Policy(handlers=lambda: [CostCalculator()])]
        if policies is not None:
            scope_policies.extend(policies)
        self._session_scope = SessionScope(
            model=model,
            stream=stream,
            max_steps=max_steps,
            policies=scope_policies,
            persistence=persistence,
        )

    @property
    def input_tool(self) -> Input:
        return self._input_tool

    @property
    def output_tool(self) -> Output:
        return self._output_tool

    def create_session(self) -> str:
        """Create a new active CLI session and return its ID."""
        return self._session_manager.create_new_active_session()

    def switch_session(self, session_id: str) -> str:
        """Switch the active CLI session and return its ID."""
        try:
            return self._session_manager.switch_active_session(session_id)
        except UnknownSessionError as e:
            raise CLIUnknownSessionError(e.session_id) from None

    @asynccontextmanager
    async def session(
        self,
        *,
        session_id: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        policies: list[Policy] | None = None,
    ) -> AsyncGenerator[SefiaCLISession]:
        """Run code within a resolved Sefia CLI session context."""
        try:
            resolved_session = self._session_manager.resolve_session(session_id)
        except UnknownSessionError as e:
            raise CLIUnknownSessionError(e.session_id) from None

        try:
            await self._reporting.session_resolved(resolved_session)
            async with self._session_scope.session(
                session_id=resolved_session.session_id,
                model=model,
                stream=stream,
                policies=policies,
            ):
                try:
                    yield SefiaCLISession(channel=get_interaction_channel())
                except InferenceError as e:
                    await self._reporting.inference_error(e)
                    raise
                except InteractionRequired as pause:
                    await self._reporting.interaction_request(
                        InteractionRequest(pause.interaction_id, pause.request)
                    )
                    await self._reporting.interrupted(resolved_session)
                    raise
                except PauseException:
                    # Any pause (InteractionRequired, or a future pause type) is a
                    # graceful interrupt, not a failure. The session context
                    # is still alive here, so reporters may read running
                    # state (e.g. cost) via get_state().
                    await self._reporting.interrupted(resolved_session)
                    raise
                else:
                    await self._reporting.session_finished()
        except InferenceError:
            raise typer.Exit(code=1) from None
        except PauseException:
            raise typer.Exit(code=0)

    @staticmethod
    def _resolve_reporter(reporter: CLIReporter | None | object) -> CLIReporter | None:
        if reporter is _USE_DEFAULT_REPORTER:
            return CostReportingCLIReporter()
        return cast(CLIReporter | None, reporter)
