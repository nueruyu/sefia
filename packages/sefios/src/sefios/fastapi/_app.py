from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.responses import StreamingResponse
from sefia import Policy
from sefia_fastapi import (
    HumanInputCoordinator,
    HumanInputReceiver,
    InputRequired,
    SessionEventBroker,
    TokenEventPublisher,
    session_event_response,
)
from sefia_fastapi import UnknownSessionError as HTTPUnknownSessionError

from .._scope import SessionScope
from .._session_state import get_session_storage
from ..exceptions import NeedsInput
from ..handlers import CostCalculator
from ..policies import CustomPolicy
from ..sessions import SessionManager
from ..tools import HumanInputRequest, HumanInputResult, HumanInputTool


class SefiaHTTPSession:
    """Operations available inside a Sefia HTTP session context."""

    def __init__(self, *, human_input: HumanInputReceiver):
        self._human_input = human_input

    async def accept_input(
        self,
        input_value: str | list[str] | None,
        *,
        reply_to: str | None = None,
    ) -> None:
        """Store request input as an answer for a pending or upcoming interaction."""
        await self._human_input.receive_input(input_value, reply_to=reply_to)


class SefiaHTTP:
    """Creates Sefia session contexts for HTTP endpoints, with event streams.

    The integration facade over the ``sefia_fastapi`` building blocks: it
    wires the HTTP human-input core to :class:`HumanInputTool` and the bound
    session storage, runs sessions through a :class:`SessionScope` (with cost
    accounting installed), relays LLM tokens to per-session SSE streams, and
    surfaces pauses as :class:`sefia_fastapi.InputRequired`.
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        model: str | None = None,
        max_steps: int | None = 25,
        policies: list[Policy] | None = None,
    ):
        self._events = SessionEventBroker()
        self._session_manager = SessionManager(session_dir)
        self._human_input = HumanInputCoordinator()
        self._human_input_receiver = HumanInputReceiver(self._human_input.store)
        self._human_input_tool = HumanInputTool(
            get_answer=self._provide_answer,
            on_request=self._record_request,
            on_complete=self._complete_request,
        )

        scope_policies: list[Policy] = [
            CustomPolicy(handlers=lambda: [CostCalculator()])
        ]
        if policies is not None:
            scope_policies.extend(policies)

        self._session_scope = SessionScope(
            session_dir=session_dir,
            model=model,
            stream=False,
            max_steps=max_steps,
            policies=scope_policies,
        )

    @property
    def human_input_tool(self) -> HumanInputTool:
        return self._human_input_tool

    def create_session(self) -> str:
        return self._session_manager.create_new_active_session()

    def ensure_session(self, session_id: str) -> None:
        if not self._session_manager.session_exists(session_id):
            raise HTTPUnknownSessionError(session_id)

    @asynccontextmanager
    async def session(
        self,
        *,
        session_id: str,
        model: str | None = None,
        stream: bool | None = None,
        policies: list[Policy] | None = None,
    ) -> AsyncIterator[SefiaHTTPSession]:
        self.ensure_session(session_id)
        # Always publish token events. Publishing is a no-op when nobody is
        # subscribed, and keeping streaming on unconditionally means a client that
        # subscribes mid-run still receives the remaining tokens -- otherwise
        # streaming would depend on subscribing strictly before the request that
        # starts the run.
        resolved_stream = True if stream is None else stream
        session_policies: list[Policy] = list(policies or [])
        if resolved_stream:
            session_policies.append(
                CustomPolicy(
                    handlers=lambda: [TokenEventPublisher(self._events, session_id)]
                )
            )

        paused_request: dict | None = None
        try:
            async with self._session_scope.session(
                session_id=session_id,
                model=model,
                stream=resolved_stream,
                policies=session_policies or None,
            ):
                with self._human_input.store.use_store(get_session_storage()):
                    try:
                        yield SefiaHTTPSession(human_input=self._human_input_receiver)
                    except NeedsInput:
                        # Read the pending request while the session store is
                        # still bound, so no second session scope is needed, then
                        # re-raise so the session scope handles the pause exactly
                        # as before. InputRequired is raised only after the scope
                        # exits, both to keep glyff's pause/resume semantics and
                        # because it is a frozen dataclass that cannot carry the
                        # traceback the scope's exit would set on it.
                        pending = await self._human_input.store.pending_requests()
                        if not pending:
                            raise RuntimeError(
                                "NeedsInput raised but no pending input request found."
                            ) from None
                        paused_request = next(iter(sorted(pending.items())))[1]
                        raise
        except NeedsInput:
            request = paused_request
            assert request is not None  # set before the re-raise above
            await self._events.publish(
                session_id,
                "input_required",
                {
                    "interaction_id": request["id"],
                    "question": request["question"],
                },
            )
            raise InputRequired(
                interaction_id=request["id"],
                question=request["question"],
            ) from None
        except Exception as exc:
            await self._events.publish(
                session_id,
                "error",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise
        else:
            await self._events.publish(
                session_id, "completed", {"session_id": session_id}
            )

    def events(self, session_id: str) -> StreamingResponse:
        self.ensure_session(session_id)
        return session_event_response(self._events, session_id)

    async def _provide_answer(self, request: HumanInputRequest) -> str | None:
        return await self._human_input.provide_answer(request.interaction_id)

    async def _record_request(self, request: HumanInputRequest) -> None:
        await self._human_input.record_request(request.interaction_id, request.question)

    async def _complete_request(self, result: HumanInputResult) -> None:
        await self._human_input.complete_request(result.interaction_id)
