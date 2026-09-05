"""Run documentation examples with real sessions and mocked external services."""

import ast
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sefia.testing import MockLLMClient, result_completion, tool_calls_completion
from sefia_litellm import LiteLLMClient
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = "docs/tutorial.md"
CLI_SECTION = "3. Make it pause for a human - and survive a restart"
REPORT: dict[str, str | list[str]] = {
    "topic": "durable execution",
    "summary": "Approved report",
    "sources": [],
}


class HTTPClient(Protocol):
    def post(self, url: str, *, json: object | None = None) -> Response: ...


def python_blocks(markdown: str) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", markdown, re.M | re.S)


def example(path: str, section: str) -> str:
    markdown = (ROOT / path).read_text(encoding="utf-8")
    _, heading, body = markdown.partition(f"## {section}\n")
    assert heading, f"Missing section {section!r} in {path}"
    return python_blocks(body.split("\n## ", 1)[0])[0]


def load_code(code: str, name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType(name)
    # Type-hint resolution and persisted results need an importable module name.
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(code, name, "exec"), module.__dict__)
    return module


@pytest.fixture(autouse=True)
def isolated_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> MockLLMClient:
    client = MockLLMClient([])
    monkeypatch.setattr(LiteLLMClient, "complete", client.complete)
    return client


def test_python_blocks_compile() -> None:
    for path in ROOT.rglob("*.md"):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        for index, code in enumerate(python_blocks(path.read_text(encoding="utf-8"))):
            compile(
                code, f"{path}:{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
            )


def test_tutorial_quickstart(
    llm: MockLLMClient,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    llm.completions.append(
        result_completion({"key_points": ["Durable calls"], "uncertainty": "None"})
    )
    code = example(TUTORIAL, "1. Your first inferred function")
    load_code(code, "__main__", monkeypatch)
    assert "Durable calls" in capsys.readouterr().out
    assert len(llm.requests) == 1


def test_tutorial_cli_pause_resume(
    llm: MockLLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm.completions.extend(
        [
            tool_calls_completion(("Input_get_input", {"prompt": "Approve draft?"})),
            result_completion(REPORT),
        ]
    )
    code = example(TUTORIAL, CLI_SECTION)
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
    ("path", "section"),
    [
        ("README.md", "Pause for a human, resume after a restart"),
        (TUTORIAL, "4. Serve it over HTTP"),
    ],
)
def test_http_pause_resume(
    path: str, section: str, llm: MockLLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm.completions.extend(
        [
            tool_calls_completion(("Input_get_input", {"prompt": "Approve draft?"})),
            result_completion(REPORT),
        ]
    )
    if path == TUTORIAL:
        load_code(example(TUTORIAL, CLI_SECTION), "hitl_cli", monkeypatch)
    code = example(path, section)
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


@pytest.mark.parametrize("path", ["README.md", TUTORIAL])
async def test_research_tool(
    path: str,
    llm: MockLLMClient,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if path == "README.md":
        code = example(path, "Quickstart")
    else:
        code = example(path, "1. Your first inferred function")
        code += "\n" + example(path, "2. Give it a tool")
    module = load_code(code, "doc_research", monkeypatch)
    llm.completions.extend(
        [
            tool_calls_completion(("WebSearch_search", {"query": "durable execution"})),
            result_completion(REPORT),
        ]
    )
    with patch("ddgs.DDGS") as search_factory:
        search = search_factory.return_value.__enter__.return_value.text
        search.return_value = [
            {"title": "Example", "href": "https://example.com", "body": "Source"}
        ]
        if path == "README.md":
            report = await module.main("durable execution")
            assert report.summary == REPORT["summary"]
        else:
            await module.main()
            assert "Approved report" in capsys.readouterr().out
        search.assert_called_once_with("durable execution", max_results=5)
    assert len(llm.requests) == 2
    assert not llm.completions
    assert "Error executing tool" not in str(llm.requests[-1]["messages"])
