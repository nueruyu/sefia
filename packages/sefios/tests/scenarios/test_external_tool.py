from pathlib import Path

import pytest
from pydantic import BaseModel
from sefia import Policy, Tools, current_tool_call_id
from sefia.event_system import EventHandler
from sefia.events import AfterToolCall, BeforeToolCall
from sefia.testing import MockLLMClient, result_completion, tool_calls_completion
from sefios import FilePersistence, SQLitePersistence, domain, require_interaction
from sefios.exceptions import InteractionRequired
from sefios.fastapi import SefiaHTTP
from sefios.interactions import InteractionRequest, JsonValue
from typing_extensions import override

infer = domain(__name__).infer


class WeatherResult(BaseModel):
    temperature: int


class ToolEvents(EventHandler[BeforeToolCall | AfterToolCall]):
    def __init__(self):
        self.events: list[BeforeToolCall | AfterToolCall] = []

    @override
    async def handle(self, event: BeforeToolCall | AfterToolCall) -> None:
        self.events.append(event)


class Weather:
    async def get_weather(self, city: str) -> WeatherResult:
        interaction_id = current_tool_call_id()
        assert interaction_id is not None
        return await require_interaction(
            interaction_id,
            {"type": "weather", "arguments": {"city": city}},
            WeatherResult,
        )


class Forecast:
    weather: Tools[Weather]

    def __init__(self):
        self.weather = Weather()

    @infer
    async def run(self) -> str: ...


@pytest.mark.parametrize("backend", ["sqlite", "file"])
async def test_arbitrary_model_tool_replays_after_restart(tmp_path: Path, backend: str):
    events = ToolEvents()

    def app(client: MockLLMClient) -> SefiaHTTP:
        persistence = (
            SQLitePersistence(tmp_path / "db")
            if backend == "sqlite"
            else FilePersistence(tmp_path)
        )
        return SefiaHTTP(
            llm_client=client,
            persistence=persistence,
            policies=[Policy(handlers=lambda: [events])],
        )

    first = app(
        MockLLMClient(
            [tool_calls_completion(("Weather_get_weather", {"city": "Tokyo"}))]
        )
    )
    sid = first.create_session()
    with pytest.raises(InteractionRequired) as pause:
        async with first.session(session_id=sid):
            await Forecast().run()
    interaction_id = pause.value.interaction_id
    request: JsonValue = {"type": "weather", "arguments": {"city": "Tokyo"}}
    assert pause.value.request == request
    assert events.events[0].tool_call.id == interaction_id

    second = app(MockLLMClient([]))
    async with second.session(session_id=sid) as session:
        assert await session.pending_interactions() == [
            InteractionRequest(interaction_id, request)
        ]
        await session.resolve_interaction(interaction_id, {"temperature": 21})
        await session.resolve_interaction(interaction_id, {"temperature": 21})
        assert await session.pending_interactions() == []

    resumed_client = MockLLMClient([result_completion("21 degrees")])
    async with app(resumed_client).session(session_id=sid):
        assert await Forecast().run() == "21 degrees"
    assert len(resumed_client.requests) == 1
    assert "21" in str(resumed_client.requests[0])
    assert [e.tool_call.id for e in events.events] == [interaction_id] * 3
    completed = events.events[-1]
    assert isinstance(completed, AfterToolCall)
    assert isinstance(completed.result, WeatherResult)
    assert completed.result.temperature == 21
