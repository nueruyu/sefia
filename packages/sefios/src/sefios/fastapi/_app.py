from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar

from fastapi.responses import StreamingResponse
from sefia import Policy
from sefia.exceptions import PauseException
from sefia.llm import LLMClient
from sefia_fastapi.events import SessionEvents, SSEEvent
from sefia_fastapi.exceptions import UnknownSessionError as HTTPUnknownSessionError
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
from ..sessions import SessionRegistry
from ..tools import Input, Output, OutputMessage


@final
class SefiaHTTPSession:
    """Operations available inside a Sefia HTTP session context."""

    def __init__(self, *, channel: InteractionChannel):
        self._channel = channel

    async def resolve_interaction(self, interaction_id: str, result: JsonValue) -> None:
        """Resolve an existing interaction with an immutable JSON result."""
        await self._channel.resolve(interaction_id, result)

    async def pending_interactions(self) -> list[InteractionRequest]:
        return await self._channel.pending()


@final
class SefiaHTTP:
    """Creates Sefia session contexts for HTTP endpoints, with event streams.

    The integration facade over the ``sefia_fastapi`` building blocks: it
    binds generic interactions and :class:`Input` previews to the session,
    runs sessions through a :class:`SessionScope` with cost accounting,
    forwards the parsed prompt/message deltas to
    per-session SSE streams, and surfaces pauses as
    :class:`~sefios.exceptions.InteractionRequired`.

    Pass ``llm_client`` to use a custom :class:`~sefia.llm.LLMClient` instead
    of constructing the default LiteLLM-backed client from ``model``.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        llm_client: LLMClient | None = None,
        max_steps: int | None = 25,
        policies: list[Policy] | None = None,
        persistence: PersistenceProvider | None = None,
    ):
        persistence = persistence or MemoryPersistence()
        self._events = SessionEvents()
        self._session_registry: SessionRegistry = persistence.create_session_registry()
        self._active_session_id: ContextVar[str | None] = ContextVar(
            "http_active_session_id", default=None
        )
        self._input_tool = Input(
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
            model=model,
            llm_client=llm_client,
            stream=True,
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
        return self._session_registry.create_session()

    def ensure_session(self, session_id: str) -> None:
        if not self._session_registry.session_exists(session_id):
            raise HTTPUnknownSessionError(session_id)

    @asynccontextmanager
    async def session(
        self,
        *,
        session_id: str,
        model: str | None = None,
        stream: bool | None = None,
        policies: list[Policy] | None = None,
    ) -> AsyncGenerator[SefiaHTTPSession]:
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
                yield SefiaHTTPSession(channel=get_interaction_channel())
        except InteractionRequired as pause:
            await self._events.publish(
                session_id,
                SSEEvent.INTERACTION_REQUIRED,
                {
                    "interaction_id": pause.interaction_id,
                    "request": pause.request,
                },
            )
            raise
        except PauseException:
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
