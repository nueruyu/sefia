import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest
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
