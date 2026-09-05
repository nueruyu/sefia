"""Exercise the documented programs with scripted model and search responses."""

import ast
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sefia.testing import MockLLMClient, result_completion, tool_calls_completion
from sefia_litellm import LiteLLMClient
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
REPORT: dict[str, Any] = {"topic": "durable execution", "summary": "Approved report", "sources": []}


class HTTPClient(Protocol):
    def post(self, url: str, *, json: object | None = None) -> Response: ...


def blocks(path: str) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", (ROOT / path).read_text(), re.M | re.S)


def load_code(code: str, name: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(code, name, "exec"), module.__dict__)
    return module


@pytest.fixture
def isolated_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def script(monkeypatch: pytest.MonkeyPatch, *, pause: bool = False) -> MockLLMClient:
    completions = [result_completion(REPORT)]
    if pause:
        completions.insert(
            0, tool_calls_completion(("Input_get_input", {"prompt": "Approve draft?"}))
        )
    llm = MockLLMClient(completions)
    monkeypatch.setattr(LiteLLMClient, "complete", llm.complete)
    return llm


def test_python_blocks_compile() -> None:
    for path in ROOT.rglob("*.md"):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        for index, code in enumerate(blocks(str(path.relative_to(ROOT)))):
            compile(
                code, f"{path}:{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
            )


@pytest.mark.usefixtures("isolated_workdir")
def test_tutorial_quickstart(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = MockLLMClient(
        [result_completion({"key_points": ["Durable calls"], "uncertainty": "None"})]
    )
    monkeypatch.setattr(LiteLLMClient, "complete", llm.complete)
    load_code(blocks("docs/tutorial.md")[0], "__main__", monkeypatch)
    assert len(llm.requests) == 1


@pytest.mark.usefixtures("isolated_workdir")
def test_tutorial_cli_pause_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = script(monkeypatch, pause=True)
    code = next(c for c in blocks("docs/tutorial.md") if "# hitl_cli.py" in c)
    first = load_code(code, "hitl_cli", monkeypatch)
    paused = CliRunner().invoke(first.app, [])
    assert paused.exit_code == 0, paused.output
    assert "DONE:" not in paused.output
    assert len(llm.requests) == 1
    session_id = first.cli.get_active_session()
    assert session_id is not None

    resumed = load_code(code, "hitl_cli", monkeypatch)
    assert resumed.cli.get_active_session() == session_id
    done = CliRunner().invoke(resumed.app, ["--answer", "yes, approve"])
    assert done.exit_code == 0, done.output
    assert "DONE: Approved report" in done.output
    assert len(llm.requests) == 2
    assert not llm.completions


@pytest.mark.parametrize(
    "path", ["README.md", "docs/tutorial.md", "docs/usecases/01-human-in-the-loop.md"]
)
@pytest.mark.usefixtures("isolated_workdir")
def test_http_pause_resume(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = script(monkeypatch, pause=True)
    if path == "docs/tutorial.md":
        cli_code = next(c for c in blocks(path) if "# hitl_cli.py" in c)
        load_code(cli_code, "hitl_cli", monkeypatch)
    code = next(c for c in blocks(path) if "app = FastAPI()" in c)
    first = load_code(code, "doc_server", monkeypatch)
    with TestClient(first.app) as raw_client:
        client = cast(HTTPClient, raw_client)
        session_id = client.post("/sessions").json()["session_id"]
        url = f"/sessions/{session_id}/turn"
        paused = client.post(url, json={"task": "durable execution"})
        assert paused.status_code == 200
        assert paused.json() == {"status": "needs_input", "prompt": "Approve draft?"}
    assert len(llm.requests) == 1

    resumed = load_code(code, "doc_server", monkeypatch)
    with TestClient(resumed.app) as raw_client:
        client = cast(HTTPClient, raw_client)
        done = client.post(
            url, json={"task": "durable execution", "input": "yes, approve"}
        )
        assert done.status_code == 200
        assert done.json()["status"] == "done"
        assert done.json()["report"]["summary"] == REPORT["summary"]
    assert len(llm.requests) == 2
    assert not llm.completions


@pytest.mark.parametrize("path", ["README.md", "docs/tutorial.md"])
@pytest.mark.usefixtures("isolated_workdir")
async def test_research_tool(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if path == "README.md":
        code = next(c for c in blocks(path) if "async def main(topic:" in c)
    else:
        first, tools = blocks(path)[:2]
        code = first.split('if __name__ == "__main__":')[0] + tools
    module = load_code(code, "doc_research", monkeypatch)
    llm = MockLLMClient(
        [
            tool_calls_completion(("WebSearch_search", {"query": "durable execution"})),
            result_completion(REPORT),
        ]
    )
    monkeypatch.setattr(LiteLLMClient, "complete", llm.complete)
    calls: list[str] = []

    class Search:
        def __enter__(self) -> "Search":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def text(self, query: str, *, max_results: int) -> list[dict[str, str]]:
            calls.append(query)
            return [
                {"title": "Example", "href": "https://example.com", "body": "Source"}
            ]

    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", Search)
    if path == "README.md":
        report = await module.main("durable execution")
        assert report.summary == REPORT["summary"]
    else:
        await module.main()
    assert calls == ["durable execution"]
    assert len(llm.requests) == 2
    assert "Error executing tool" not in str(llm.requests)
