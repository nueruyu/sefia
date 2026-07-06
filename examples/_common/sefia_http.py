from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from fastapi.responses import StreamingResponse
from sefia.exceptions import NeedsInput
from sefia import Policy
from sefia.event_system import EventHandler
from sefia.llm.events import LLMTokenReceived
from sefios import SessionScope, get_session_state
from sefios.handlers import CostCalculator
from sefios.policies import CustomPolicy
from sefios.tools import HumanInputTool

from .human_input import CLIHumanInputAdapter, CLIHumanInputReceiver
from .session import SessionManager, UnknownSessionError


@dataclass(frozen=True)
class InputRequired(Exception):
    """Raised when a session pauses to wait for external input."""

    interaction_id: str
    question: str

    def __str__(self) -> str:
        return f"Input required: {self.question}"


@dataclass(frozen=True)
class _SessionEvent:
    name: str
    data: Any


class _SessionEventBroker:
    def __init__(self):
        self._subscribers: dict[str, set[asyncio.Queue[_SessionEvent]]] = {}

    @asynccontextmanager
    async def subscribe(
        self, session_id: str
    ) -> AsyncIterator[asyncio.Queue[_SessionEvent]]:
        queue: asyncio.Queue[_SessionEvent] = asyncio.Queue()
        subscribers = self._subscribers.setdefault(session_id, set())
        subscribers.add(queue)
        try:
            yield queue
        finally:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(session_id, None)

    async def publish(self, session_id: str, name: str, data: Any) -> None:
        subscribers = list(self._subscribers.get(session_id, ()))
        if not subscribers:
            return
        event = _SessionEvent(name=name, data=data)
        for queue in subscribers:
            await queue.put(event)


class _TokenEventPublisher(EventHandler[LLMTokenReceived]):
    def __init__(self, broker: _SessionEventBroker, session_id: str):
        self._broker = broker
        self._session_id = session_id

    async def handle(self, event: LLMTokenReceived) -> None:
        await self._broker.publish(self._session_id, "token", event.token)


class SefiaHTTPSession:
    """Operations available inside a Sefia HTTP session context."""

    def __init__(self, *, human_input: CLIHumanInputReceiver):
        self._human_input = human_input

    async def accept_input(
        self,
        input_value: str | list[str] | None,
        *,
        reply_to: str | None = None,
    ) -> None:
        if input_value is None:
            return
        input_text = _to_input_text(input_value)
        if not input_text:
            return
        await self._human_input.receive_input(input_text, reply_to=reply_to)


class SefiaHTTP:
    """Example-local HTTP helper for normal endpoints plus session event streams."""

    def __init__(
        self,
        *,
        session_dir: Path,
        model: str | None = None,
        human_input_adapter: CLIHumanInputAdapter | None = None,
        max_steps: int | None = 25,
        policies: list[Policy] | None = None,
    ):
        self._events = _SessionEventBroker()
        self._session_manager = SessionManager(session_dir)
        self._human_input = human_input_adapter or CLIHumanInputAdapter()
        self._human_input_tool = self._human_input.create_tool()
        self._human_input_receiver = CLIHumanInputReceiver(self._human_input.store)

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
            raise UnknownSessionError(session_id)

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
                    handlers=lambda: [_TokenEventPublisher(self._events, session_id)]
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
                with self._human_input.store.use_session_store(
                    get_session_state().store
                ):
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
        return StreamingResponse(
            self._event_stream(session_id),
            media_type="text/event-stream",
        )

    async def _event_stream(self, session_id: str) -> AsyncIterator[str]:
        async with self._events.subscribe(session_id) as queue:
            while True:
                event = await queue.get()
                yield _sse_event(event.name, event.data)


def _to_input_text(input_value: str | list[str]) -> str:
    if isinstance(input_value, str):
        return input_value.strip()
    return " ".join(input_value).strip()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _sse_event(event: str, data: Any) -> str:
    payload = json.dumps(_jsonable(data), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
