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
REPORT: dict[str, str | list[str]] = {
    "topic": "durable execution",
    "summary": "Approved report",
    "sources": [],
}


class HTTPClient(Protocol):
    def post(self, url: str, *, json: object | None = None) -> Response: ...


def python_blocks(markdown: str) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", markdown, re.M | re.S)


def example(path: str, example_id: str) -> str:
    markdown = (ROOT / path).read_text(encoding="utf-8")
    marker = rf"^<!-- example: {re.escape(example_id)} -->[ \t]*$"
    markers = list(re.finditer(marker, markdown, re.M))
    location = f"{path}: example {example_id!r}"
    assert len(markers) == 1, f"{location}: expected one marker, found {len(markers)}"
    block = re.match(
        r"\s*```python[ \t]*\n(.*?)^```[ \t]*(?:\n|$)",
        markdown[markers[0].end() :],
        re.M | re.S,
    )
    assert block is not None, f"{location}: expected Python block immediately after ID"
    return block[1]


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


@pytest.mark.parametrize("target_first", [True, False])
def test_example_selection_ignores_layout(tmp_path: Path, target_first: bool) -> None:
    target = "<!-- example: chosen -->\n```python\nanswer = 42\n```\n"
    other = "## Another heading\n```python\nanswer = 0\n```\n"
    parts = [target, other] if target_first else [other, target]
    path = tmp_path / "examples.md"
    path.write_text("## Renamed heading\n" + "\n".join(parts), encoding="utf-8")
    assert example(str(path), "chosen") == "answer = 42\n"


@pytest.mark.parametrize(
    ("markdown", "error"),
    [
        ("<!-- example: other -->\n```python\npass\n```", "found 0"),
        ("<!-- example: chosen -->\n" * 2, "found 2"),
        ("<!-- example: chosen -->\n```bash\necho hi\n```", "Python block"),
        ("<!-- example: chosen -->\nprose\n```python\npass\n```", "Python block"),
        ("<!-- example: chosen -->\n```python\npass", "Python block"),
    ],
)
def test_example_rejects_invalid_marker(
    tmp_path: Path, markdown: str, error: str
) -> None:
    path = tmp_path / "examples.md"
    path.write_text(markdown, encoding="utf-8")
    with pytest.raises(AssertionError, match=error) as caught:
        example(str(path), "chosen")
    assert str(path) in str(caught.value)
    assert "chosen" in str(caught.value)


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
    code = example(TUTORIAL, "tutorial-quickstart")
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
    code = example(TUTORIAL, "tutorial-cli")
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


def test_readme_http_excerpt_matches_tutorial() -> None:
    excerpt = ast.parse(example("README.md", "readme-http"))
    tutorial_nodes = {
        ast.dump(node)
        for example_id in ("tutorial-cli", "tutorial-http")
        for node in ast.parse(example(TUTORIAL, example_id)).body
    }
    assert excerpt.body
    for node in excerpt.body:
        assert ast.dump(node) in tutorial_nodes



def test_http_pause_resume(llm: MockLLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
    llm.completions.extend(
        [
            tool_calls_completion(("Input_get_input", {"prompt": "Approve draft?"})),
            result_completion(REPORT),
        ]
    )
    load_code(example(TUTORIAL, "tutorial-cli"), "hitl_cli", monkeypatch)
    code = example(TUTORIAL, "tutorial-http")
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
        code = example(path, "readme-quickstart")
    else:
        code = example(path, "tutorial-quickstart")
        code += "\n" + example(path, "tutorial-tools")
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
