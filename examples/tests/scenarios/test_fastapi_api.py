"""Scenario tests for the FastAPI example.

No real LLM calls are made: the agents' ``@infer`` methods are replaced while
session persistence and the pause/resume path run for real.
"""

from importlib import import_module
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from examples._common.sefia_http import SefiaHTTP

app_module = import_module("examples.03_fastapi_api.app")
agents_module = import_module("examples.03_fastapi_api.agents")
domain_module = import_module("examples.03_fastapi_api.models")

Brief = domain_module.Brief


@pytest.fixture
def api(tmp_path):
    service = SefiaHTTP(session_dir=tmp_path / "sessions", model="gpt-4o-mini")
    app = app_module.create_app(service)
    client = TestClient(app)
    return SimpleNamespace(client=client, service=service, app=app)


def _new_session(client: TestClient) -> str:
    response = client.post("/sessions")
    assert response.status_code == 200
    return response.json()["session_id"]


def test_index_serves_browser_ui(api):
    response = api.client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Sefia FastAPI Example" in response.text
    assert "chat-log" in response.text
    assert "new EventSource" in response.text
    assert "appendToken" in response.text


class TestAnswerEndpoint:
    def test_returns_completed_answer(self, api, monkeypatch):
        async def fake_answer(self, question: str) -> str:
            return "A vector store."

        monkeypatch.setattr(agents_module.Assistant, "answer", fake_answer)
        session_id = _new_session(api.client)

        response = api.client.post(
            f"/sessions/{session_id}/answer", json={"question": "What is it?"}
        )

        assert response.status_code == 200
        assert response.json() == {"status": "completed", "answer": "A vector store."}

    def test_unknown_session_is_404(self, api, monkeypatch):
        async def fake_answer(self, question: str) -> str:
            return "ignored"

        monkeypatch.setattr(agents_module.Assistant, "answer", fake_answer)
        response = api.client.post(
            "/sessions/does-not-exist/answer", json={"question": "hi"}
        )
        assert response.status_code == 404

    def test_unknown_session_events_is_404(self, api):
        response = api.client.get("/sessions/does-not-exist/events")
        assert response.status_code == 404


class TestInterviewFlow:
    def test_pauses_then_resumes_to_completion(self, api, monkeypatch):
        question = "Who is the target audience?"
        tool = api.service.human_input_tool

        async def fake_run(self):
            topic = await tool.get_human_input("What should this be about?")
            audience = await tool.get_human_input(question)
            return Brief(topic=topic, goal="Inform", audience=audience)

        monkeypatch.setattr(agents_module.Interviewer, "run", fake_run)
        session_id = _new_session(api.client)

        first = api.client.post(
            f"/sessions/{session_id}/interview",
            json={"input": "Write about our product."},
        )
        assert first.status_code == 200
        first_body = first.json()
        assert first_body["status"] == "input_required"
        assert first_body["question"] == question
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
