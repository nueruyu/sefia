import json
import os
import shlex
import subprocess
import sys
import textwrap
from importlib import metadata
from pathlib import Path

import pytest
from package_compatibility import validation
from package_compatibility.__main__ import main


def test_json_plan(manifests: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "--root", str(manifests), "--all", "--version", "0.6.1"]) == 0
    captured = capsys.readouterr()
    plan = json.loads(captured.out)
    assert plan["candidate_version"] == "0.6.1"
    assert len(plan["matrix"]["include"]) == 5
    assert plan["has_packages"] is True
    assert "explicit intended release" in captured.err


@pytest.mark.parametrize("version", ["not-a-version", "0.7.0"])
def test_invalid_version_fails_before_build(
    manifests: Path, tmp_path: Path, version: str, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "dist"
    assert (
        main(
            [
                "build",
                "--root",
                str(manifests),
                "--version",
                version,
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()
    assert "failed" in capsys.readouterr().err


def test_missing_root_and_selection(tmp_path: Path, manifests: Path) -> None:
    assert main(["plan", "--root", str(tmp_path / "missing"), "--all"]) == 1
    assert main(["plan", "--root", str(manifests)]) == 1


def test_installed_entrypoints_outside_checkout(manifests: Path) -> None:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GITHUB_")
    }
    environment.pop("PYTHONPATH", None)
    assert metadata.version("sefia-package-compatibility") == "0.1.0"
    console = Path(sys.executable).parent / "package-compatibility"
    for entrypoint in ([sys.executable, "-m", "package_compatibility"], [str(console)]):
        subprocess.run(
            [*entrypoint, "--help"],
            env=environment,
            cwd=manifests,
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            [*entrypoint, "plan", "--root", str(manifests), "--all"],
            env=environment,
            cwd=manifests,
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(result.stdout)["has_packages"] is True


def test_test_exit_two_is_failure_not_unreleased(
    manifests: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from package_compatibility import validation

    def fail(
        root: Path,
        distribution: str,
        version: str,
        wheels: Path,
        *,
        minimum: bool = False,
    ) -> validation.MinimumStatus:
        assert root == manifests and distribution == "sefios" and version and wheels
        assert minimum
        raise subprocess.CalledProcessError(2, ["python", "-m", "pytest"])

    monkeypatch.setattr(validation, "validate", fail)
    assert (
        main(
            [
                "validate-minimum",
                "--root",
                str(manifests),
                "--distribution",
                "sefios",
                "--version",
                "0.6.0",
                "--wheels",
                str(tmp_path),
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [("available", 0), ("unreleased", 10), ("not-applicable", 11)],
)
def test_minimum_status_exit_codes(
    manifests: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: validation.MinimumStatus,
    exit_code: int,
) -> None:
    def result(
        root: Path,
        distribution: str,
        version: str,
        wheels: Path,
        *,
        minimum: bool = False,
    ) -> validation.MinimumStatus:
        assert root == manifests and distribution == "sefios" and version and wheels
        assert minimum
        return status

    monkeypatch.setattr(validation, "validate", result)
    assert (
        main(
            [
                "validate-minimum",
                "--root",
                str(manifests),
                "--distribution",
                "sefios",
                "--version",
                "0.6.0",
                "--wheels",
                str(tmp_path),
            ]
        )
        == exit_code
    )


@pytest.fixture
def minimum_workflow_script() -> str:
    workflow = (
        Path(__file__).resolve().parents[3]
        / ".github/workflows/package-compatibility.yml"
    )
    step = workflow.read_text().split("- name: Validate minimum dependencies\n", 1)[1]
    return textwrap.dedent(step.split("run: |\n", 1)[1])


@pytest.mark.parametrize("invalid", ["distribution", "missing-wheels"])
def test_workflow_rejects_cli_argument_errors(
    manifests: Path,
    tmp_path: Path,
    minimum_workflow_script: str,
    invalid: str,
) -> None:
    script = minimum_workflow_script
    if invalid == "missing-wheels":
        script = script.replace("--wheels candidate-dist", "")
    summary = tmp_path / "summary"
    environment = validation.clean_environment()
    environment.update(
        DISTRIBUTION="unknown" if invalid == "distribution" else "sefios",
        CANDIDATE_VERSION="0.6.0",
        GITHUB_STEP_SUMMARY=str(summary),
    )
    command = shlex.join([sys.executable, "-m", "package_compatibility"])
    result = subprocess.run(
        ["bash", "-c", f'uv() {{ shift 5; {command} "$@"; }}\n' + script],
        cwd=manifests,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert not summary.exists()


@pytest.mark.parametrize(
    ("exit_code", "message"),
    [
        (0, "minimum verified"),
        (10, "unreleased minimum / not verified"),
        (11, "minimum not applicable"),
    ],
)
def test_workflow_reports_minimum_status(
    manifests: Path,
    tmp_path: Path,
    minimum_workflow_script: str,
    exit_code: int,
    message: str,
) -> None:
    summary = tmp_path / "summary"
    environment = validation.clean_environment()
    environment.update(
        DISTRIBUTION="sefios",
        CANDIDATE_VERSION="0.6.0",
        GITHUB_STEP_SUMMARY=str(summary),
    )
    result = subprocess.run(
        ["bash", "-c", f"uv() {{ return {exit_code}; }}\n" + minimum_workflow_script],
        cwd=manifests,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert f"sefios: {message}" in summary.read_text()
