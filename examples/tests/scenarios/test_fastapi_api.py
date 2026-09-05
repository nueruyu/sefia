"""Scenario tests for the FastAPI example.

No real LLM calls are made: the agents' ``@infer`` methods are replaced while
session persistence and the pause/resume path run for real.
"""

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Protocol, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from sefia.testing import MockLLMClient, result_completion, tool_calls_completion
from sefios.fastapi import SefiaHTTP

app_module: ModuleType = import_module("examples.03_fastapi_api.app")
agents_module: ModuleType = import_module("examples.03_fastapi_api.agents")
domain_module: ModuleType = import_module("examples.03_fastapi_api.models")

Brief = domain_module.Brief


class _HTTPClient(Protocol):
    def get(self, url: str) -> Response: ...

    def post(self, url: str, *, json: object | None = None) -> Response: ...


@dataclass
class _API:
    client: _HTTPClient
    service: SefiaHTTP
    app: FastAPI


@pytest.fixture
def api() -> _API:
    service = SefiaHTTP(model="gpt-4o-mini")
    app = cast(FastAPI, app_module.create_app(service))
    client = cast(_HTTPClient, TestClient(app))
    return _API(client=client, service=service, app=app)


def _new_session(client: _HTTPClient) -> str:
    response = client.post("/sessions")
    assert response.status_code == 200
    return response.json()["session_id"]


def test_index_serves_hitl_browser_ui(api: _API) -> None:
    response = api.client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Sefia FastAPI HITL Example" in response.text
    assert "chat-log" in response.text
    assert "new EventSource" in response.text
    assert "/interview" in response.text
    assert "/answer" not in response.text


class TestInterviewFlow:
    def test_pauses_then_resumes_to_completion(self, api: _API) -> None:
        question = "Who is the target audience?"
        api.service._session_scope.llm_client = MockLLMClient(
            completions=[
                tool_calls_completion(
                    ("Input_get_input", {"prompt": "What should this be about?"}),
                ),
                tool_calls_completion(
                    ("Input_get_input", {"prompt": question}),
                ),
                result_completion(
                    Brief(
                        topic="Write about our product.",
                        goal="Inform",
                        audience="Developers",
                    )
                ),
            ]
        )
        session_id = _new_session(api.client)

        first = api.client.post(
            f"/sessions/{session_id}/interview",
            json={"input": "Write about our product."},
        )
        assert first.status_code == 200
        first_body = first.json()
        assert first_body["status"] == "input_required"
        assert first_body["prompt"] == question
        interaction_id = first_body["interaction_id"]

        second = api.client.post(
            f"/sessions/{session_id}/interview",
            json={"input": "Developers", "reply_to": interaction_id},
        )
        assert second.status_code == 200
        assert second.json() == {
            "status": "completed",
            "brief": {
                "topic": "Write about our product.",
                "goal": "Inform",
                "audience": "Developers",
            },
        }

    def test_unknown_session_is_404(
        self, api: _API, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_run(_self: object):
            return Brief(topic="ignored", goal="ignored", audience="ignored")

        monkeypatch.setattr(agents_module.Interviewer, "run", fake_run)
        response = api.client.post(
            "/sessions/does-not-exist/interview", json={"input": "hi"}
        )
        assert response.status_code == 404

    def test_unknown_session_events_is_404(self, api: _API) -> None:
        response = api.client.get("/sessions/does-not-exist/events")
        assert response.status_code == 404
