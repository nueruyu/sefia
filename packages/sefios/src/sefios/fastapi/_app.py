from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path

from fastapi.responses import StreamingResponse
from sefia import Policy
from sefia_fastapi import InputChannel, InputRequired, SessionEvents, SSEEvent
from sefia_fastapi import UnknownSessionError as HTTPUnknownSessionError

from .._scope import SessionScope
from .._session_state import get_session_storage
from ..exceptions import NeedsInput
from ..handlers import CostCalculator
from ..sessions import SessionManager
from ..tools import Input, InputRequest, InputResult, Output, OutputMessage


class SefiaHTTPSession:
    """Operations available inside a Sefia HTTP session context."""

    def __init__(self, *, channel: InputChannel):
        self._input = channel

    async def accept_input(
        self,
        input_value: str | list[str] | None,
        *,
        reply_to: str | None = None,
    ) -> None:
        """Store request input for a pending or upcoming interaction."""
        await self._input.receive_input(input_value, reply_to=reply_to)


class SefiaHTTP:
    """Creates Sefia session contexts for HTTP endpoints, with event streams.

    The integration facade over the ``sefia_fastapi`` building blocks: it
    wires the HTTP input core to :class:`Input` and the bound
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
        self._events = SessionEvents()
        self._session_manager = SessionManager(session_dir)
        self._input = InputChannel(namespace="http/input_channel")
        self._active_session_id: ContextVar[str | None] = ContextVar(
            "http_active_session_id", default=None
        )
        self._input_tool = Input(
            get_input=self._provide_input,
            on_request=self._record_request,
            on_complete=self._complete_request,
        )
        self._output_tool = Output(on_output=self._emit_output)

        scope_policies: list[Policy] = [Policy(handlers=lambda: [CostCalculator()])]
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
    def input_tool(self) -> Input:
        return self._input_tool

    @property
    def output_tool(self) -> Output:
        return self._output_tool

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
                Policy(handlers=lambda: [self._events.token_handler(session_id)])
            )

        token = self._active_session_id.set(session_id)
        try:
            async with self._session_scope.session(
                session_id=session_id,
                model=model,
                stream=resolved_stream,
                policies=session_policies or None,
            ):
                with self._input.use_store(get_session_storage()):
                    yield SefiaHTTPSession(channel=self._input)
        except NeedsInput as pause:
            # The pause identifies its own request, so no state is re-read.
            # InputRequired is raised only after the session scope has exited,
            # both to keep glyff's pause/resume semantics and because it is a
            # frozen dataclass that cannot carry the traceback the scope's
            # exit would set on it. A pause from a tool that did not identify
            # itself cannot be described to the HTTP client, so it propagates
            # unchanged.
            if pause.interaction_id is None:
                raise
            await self._events.publish(
                session_id,
                SSEEvent.INPUT_REQUIRED,
                {
                    "interaction_id": pause.interaction_id,
                    "prompt": pause.prompt,
                },
            )
            raise InputRequired(
                interaction_id=pause.interaction_id,
                prompt=pause.prompt,
            ) from None
        except Exception as exc:
            await self._events.publish(
                session_id,
                SSEEvent.EXECUTION_FAILED,
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise
        else:
            await self._events.publish(
                session_id, SSEEvent.COMPLETED, {"session_id": session_id}
            )
        finally:
            self._active_session_id.reset(token)

    def events(self, session_id: str) -> StreamingResponse:
        self.ensure_session(session_id)
        return self._events.response(session_id)

    async def _provide_input(self, request: InputRequest) -> str | None:
        return await self._input.provide_input(request.interaction_id)

    async def _record_request(self, request: InputRequest) -> None:
        await self._input.record_request(request.interaction_id, request.prompt)

    async def _complete_request(self, result: InputResult) -> None:
        await self._input.complete_request(result.interaction_id)

    async def _emit_output(self, message: OutputMessage) -> None:
        session_id = self._active_session_id.get()
        if session_id is None:
            raise RuntimeError(
                "Output tool is not bound to a session; send_output must run "
                "inside SefiaHTTP.session()."
            )
        await self._events.publish(
            session_id,
            SSEEvent.OUTPUT,
            {
                "interaction_id": message.interaction_id,
                "message": message.message,
            },
        )
