import inspect
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypeVar, cast

import typer
from sefia import Policy
from sefia.exceptions import InferenceError, PauseException
from sefia_typer import (
    CLIReporter,
    DefaultCLIReporter,
    InputChannel,
)
from sefia_typer import InputRequest as CLIInputRequest
from sefia_typer import ResolvedSession as CLIResolvedSession
from sefia_typer import UnknownSessionError as CLIUnknownSessionError

from .._scope import SessionScope
from .._session_state import get_session_storage
from ..handlers import CostCalculator, CostState
from ..sessions import ResolvedSession, SessionManager, UnknownSessionError
from ..state import get_state
from ..tools import Input, InputRequest, InputResult

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]
_USE_DEFAULT_REPORTER = object()


class CostReportingCLIReporter(CLIReporter):
    """Wraps a reporter to also echo the session's total cost.

    Reads the :class:`CostState` accumulated by the scope's
    :class:`CostCalculator`, so it only works inside a session run by
    :class:`SefiaCLI`. This is the default reporter of :class:`SefiaCLI`.
    """

    def __init__(self, inner: CLIReporter | None = None):
        self._inner = inner or DefaultCLIReporter()

    def on_session_resolved(self, session: CLIResolvedSession) -> MaybeAwaitable[None]:
        return self._inner.on_session_resolved(session)

    def on_input_request(self, request: CLIInputRequest) -> MaybeAwaitable[None]:
        return self._inner.on_input_request(request)

    def on_input_prompt_delta(self, text: str) -> MaybeAwaitable[None]:
        return self._inner.on_input_prompt_delta(text)

    async def on_interrupted(self, session: CLIResolvedSession) -> None:
        await _maybe_await(self._inner.on_interrupted(session))
        await self._echo_total_cost()

    async def on_inference_error(self, error: InferenceError) -> None:
        await _maybe_await(self._inner.on_inference_error(error))
        await self._echo_total_cost()

    async def on_session_finished(self) -> None:
        await _maybe_await(self._inner.on_session_finished())
        await self._echo_total_cost()

    @staticmethod
    async def _echo_total_cost() -> None:
        cost_state = await get_state().get(CostState).ensure()
        typer.echo()
        typer.secho(f"> Total cost: ${cost_state.cost:.4f}", bold=True)


class SefiaCLISession:
    """Operations available inside a Sefia CLI session context."""

    def __init__(self, *, channel: InputChannel):
        self._input = channel

    async def accept_input(
        self,
        input_value: str | list[str] | None,
        *,
        reply_to: str | None = None,
    ) -> None:
        """Store CLI input for a pending or upcoming interaction."""
        await self._input.receive_input(input_value, reply_to=reply_to)


class SefiaCLI:
    """Creates Sefia session contexts for Typer commands.

    The integration facade over the ``sefia_typer`` building blocks: it wires
    the CLI input core to :class:`Input` and the bound session
    storage, runs sessions through a :class:`SessionScope` (with cost
    accounting installed), and maps pauses and inference errors to CLI exit
    codes.
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        reporter: CLIReporter | None | object = _USE_DEFAULT_REPORTER,
        model: str | None = None,
        stream: bool = True,
        max_steps: int | None = 25,
        policies: list[Policy] | None = None,
    ):
        self._reporter = self._resolve_reporter(reporter)
        self._session_manager = SessionManager(session_dir)
        self._input = InputChannel(
            on_request=self._report_input_request,
            on_prompt_delta=self._report_input_prompt_delta,
            namespace="cli/input_channel",
        )
        self._input_tool = Input(
            get_input=self._provide_input,
            on_request=self._record_request,
            on_complete=self._complete_request,
            on_prompt_delta=self._input.notify_prompt_delta,
        )

        scope_policies: list[Policy] = [Policy(handlers=lambda: [CostCalculator()])]
        if policies is not None:
            scope_policies.extend(policies)
        self._session_scope = SessionScope(
            session_dir=session_dir,
            model=model,
            stream=stream,
            max_steps=max_steps,
            policies=scope_policies,
        )

    @property
    def input_tool(self) -> Input:
        return self._input_tool

    def create_session(self) -> str:
        """Create a new active CLI session and return its ID."""
        return self._session_manager.create_new_active_session()

    def switch_session(self, session_id: str) -> str:
        """Switch the active CLI session and return its ID."""
        try:
            return self._session_manager.switch_active_session(session_id)
        except UnknownSessionError as e:
            raise CLIUnknownSessionError(e.session_id) from None

    def get_active_session(self) -> str | None:
        """Return the active CLI session ID, if any."""
        return self._session_manager.get_active_session_id()

    @asynccontextmanager
    async def session(
        self,
        *,
        session_id: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        policies: list[Policy] | None = None,
    ) -> AsyncIterator[SefiaCLISession]:
        """Run code within a resolved Sefia CLI session context."""
        try:
            resolved_session = self._session_manager.resolve_session(session_id)
        except UnknownSessionError as e:
            raise CLIUnknownSessionError(e.session_id) from None

        try:
            await self._report_session_resolved(resolved_session)
            async with self._session_scope.session(
                session_id=resolved_session.session_id,
                model=model,
                stream=stream,
                policies=policies,
            ):
                with self._input.use_store(get_session_storage()):
                    try:
                        yield SefiaCLISession(channel=self._input)
                    except InferenceError as e:
                        await self._report_inference_error(e)
                        raise
                    except PauseException:
                        # Any pause (NeedsInput, or a future pause type) is a
                        # graceful interrupt, not a failure. The session context
                        # is still alive here, so reporters may read running
                        # state (e.g. cost) via get_state().
                        await self._report_interrupted(resolved_session)
                        raise
                    else:
                        await self._report_session_finished()
        except InferenceError:
            raise typer.Exit(code=1) from None
        except PauseException:
            raise typer.Exit(code=0)

    async def _provide_input(self, request: InputRequest) -> str | None:
        return await self._input.provide_input(request.interaction_id)

    async def _record_request(self, request: InputRequest) -> None:
        await self._input.record_request(request.interaction_id, request.prompt)

    async def _complete_request(self, result: InputResult) -> None:
        await self._input.complete_request(result.interaction_id)

    async def _report_session_resolved(self, session: ResolvedSession) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_session_resolved(session))

    async def _report_input_request(self, request: CLIInputRequest) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_input_request(request))

    async def _report_input_prompt_delta(self, text: str) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_input_prompt_delta(text))

    async def _report_interrupted(self, session: ResolvedSession) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_interrupted(session))

    async def _report_inference_error(self, error: InferenceError) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_inference_error(error))

    async def _report_session_finished(self) -> None:
        if self._reporter is not None:
            await _maybe_await(self._reporter.on_session_finished())

    @staticmethod
    def _resolve_reporter(reporter: CLIReporter | None | object) -> CLIReporter | None:
        if reporter is _USE_DEFAULT_REPORTER:
            return CostReportingCLIReporter()
        return cast(CLIReporter | None, reporter)


async def _maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value
