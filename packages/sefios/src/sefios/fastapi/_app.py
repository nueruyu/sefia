from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path

from fastapi.responses import StreamingResponse
from sefia import Policy
from sefia_fastapi import InputChannel, SessionEvents, SSEEvent
from sefia_fastapi import UnknownSessionError as HTTPUnknownSessionError

from .._scope import SessionScope
from .._session_state import get_session_storage
from ..exceptions import InputRequired
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
    accounting installed), forwards the parsed prompt/message deltas to
    per-session SSE streams, and surfaces pauses as
    :class:`~sefios.InputRequired`.
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
            on_prompt_delta=self._emit_input_delta,
        )
        self._output_tool = Output(
            on_output=self._emit_output,
            on_message_delta=self._emit_output_delta,
        )

        scope_policies: list[Policy] = [Policy(handlers=lambda: [CostCalculator()])]
        if policies is not None:
            scope_policies.extend(policies)

        self._session_scope = SessionScope(
            session_dir=session_dir,
            model=model,
            stream=True,
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
        # Stream by default: the parsed prompt/message deltas are decoded from
        # the streamed tool-call arguments, so LLM-level streaming has to be on
        # for them to flow. Publishing is a no-op when nobody is subscribed, and
        # keeping streaming on unconditionally means a client that subscribes
        # mid-run still receives the remaining deltas.
        resolved_stream = True if stream is None else stream

        token = self._active_session_id.set(session_id)
        try:
            async with self._session_scope.session(
                session_id=session_id,
                model=model,
                stream=resolved_stream,
                policies=policies,
            ):
                with self._input.use_store(get_session_storage()):
                    yield SefiaHTTPSession(channel=self._input)
        except InputRequired as pause:
            # The pause identifies its own request, so no state is re-read. The
            # SSE event is published only after the session scope has exited, to
            # keep glyff's pause/resume semantics; the pause itself then
            # propagates unchanged for the application to map to a response. A
            # pause from a tool that did not identify itself cannot be described
            # to the HTTP client, so it skips the event and propagates all the
            # same.
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
            raise
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

    async def _emit_input_delta(self, interaction_id: str, text: str) -> None:
        await self._emit_delta("input", interaction_id, text)

    async def _emit_output_delta(self, interaction_id: str, text: str) -> None:
        await self._emit_delta("output", interaction_id, text)

    async def _emit_delta(
        self, delta_type: str, interaction_id: str, text: str
    ) -> None:
        session_id = self._require_session_id()
        await self._events.publish(
            session_id,
            SSEEvent.DELTA,
            {"type": delta_type, "interaction_id": interaction_id, "text": text},
        )

    async def _emit_output(self, message: OutputMessage) -> None:
        session_id = self._require_session_id()
        await self._events.publish(
            session_id,
            SSEEvent.OUTPUT,
            {
                "interaction_id": message.interaction_id,
                "message": message.message,
            },
        )

    def _require_session_id(self) -> str:
        session_id = self._active_session_id.get()
        if session_id is None:
            raise RuntimeError(
                "The Input/Output tools are not bound to a session; they must "
                "run inside SefiaHTTP.session()."
            )
        return session_id
