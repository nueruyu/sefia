import subprocess
from importlib import import_module
from unittest.mock import AsyncMock

import pytest
from examples._common.sefia_cli import SefiaCLI

# Loaded with the full ``examples.`` prefix because main.py imports ``.._common``.
main = import_module("examples.02_code_quality.main")
models = import_module("examples.02_code_quality.models")


def _init_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


@pytest.fixture
def project(tmp_path):
    """A small tracked git project for the review workflow to operate on."""
    repo = tmp_path / "project"
    repo.mkdir()
    _init_repo(repo)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "util.py").write_text("y = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    return repo


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    cli = SefiaCLI(session_dir=tmp_path / "sessions", model="gpt-4o", stream=False)
    monkeypatch.setattr(main, "sefia_cli", cli)
    return main


def _mock_inference(workflow, monkeypatch, *, scope, review_files, issues, report):
    """Mock every @infer entry point used across the review workflow."""
    monkeypatch.setattr(
        workflow.scoping_agent, "define_scope", AsyncMock(return_value=scope)
    )
    # Keep the understanding loop to a single pass by selecting no files to read.
    monkeypatch.setattr(
        workflow.understanding_agent,
        "_prioritize_files_to_read",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        workflow.review_scoping_agent,
        "propose_and_confirm_review_files",
        AsyncMock(return_value=review_files),
    )
    review_mocks = {}
    for perspective, agent in workflow.review_agents.items():
        mock = AsyncMock(return_value=list(issues.get(perspective, [])))
        monkeypatch.setattr(agent, "review", mock)
        review_mocks[perspective] = mock
    create_report = AsyncMock(return_value=report)
    monkeypatch.setattr(workflow.reporting_agent, "create_report", create_report)
    return review_mocks, create_report


class TestCodeQualityWorkflow:
    async def test_runs_full_review_and_renders_report(
        self, workflow, project, monkeypatch, capsys
    ):
        scope = models.ProjectScope(project_path=str(project))
        perspective = models.ReviewPerspective.CODING_STYLE
        issue = models.CodeIssue(
            file_path="app.py",
            line_number=1,
            perspective="",
            description="Single-letter name.",
            suggestion="Use a descriptive name.",
        )
        report = models.QualityReport(
            overall_summary="One style issue found.",
            issues_by_perspective={perspective.value: [issue]},
        )
        review_mocks, create_report = _mock_inference(
            workflow,
            monkeypatch,
            scope=scope,
            review_files=["app.py"],
            issues={perspective: [issue]},
            report=report,
        )

        await workflow._chat_async(
            message=["Review my project"],
            reply_to=None,
            session_id=None,
            model="gpt-4o",
            verbose=False,
        )

        # Every review perspective is consulted, and the final report is rendered.
        for mock in review_mocks.values():
            mock.assert_awaited_once()
        create_report.assert_awaited_once()
        # The reviewed file's content was read and handed to each reviewer.
        coding_style_call = review_mocks[perspective].await_args.args[0]
        assert coding_style_call == {"app.py": "x = 1\n"}
        # The perspective is stamped onto each issue before reporting.
        assert issue.perspective == perspective.value

        output = capsys.readouterr().out
        assert "One style issue found." in output

    async def test_no_selected_files_skips_review(
        self, workflow, project, monkeypatch, capsys
    ):
        scope = models.ProjectScope(project_path=str(project))
        report = models.QualityReport(overall_summary="unused")
        review_mocks, create_report = _mock_inference(
            workflow,
            monkeypatch,
            scope=scope,
            review_files=[],
            issues={},
            report=report,
        )

        await workflow._chat_async(
            message=["Review my project"],
            reply_to=None,
            session_id=None,
            model="gpt-4o",
            verbose=False,
        )

        # With nothing to review, reviewers and the reporter are never invoked.
        for mock in review_mocks.values():
            mock.assert_not_awaited()
        create_report.assert_not_awaited()
        assert "No files selected for review" in capsys.readouterr().out
