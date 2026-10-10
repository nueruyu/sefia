from pathlib import Path
import json
import subprocess
import sys

import pytest
from package_compatibility import dependencies, validation


@pytest.fixture
def wheels(tmp_path: Path) -> Path:
    directory = tmp_path / "wheels"
    directory.mkdir()
    for name in dependencies.PACKAGE_ORDER:
        (directory / f"{name.replace('-', '_')}-0.6.0-py3-none-any.whl").touch()
    return directory


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def record(command: list[str], root: Path, environment: dict[str, str]) -> None:
        assert root.is_dir()
        assert environment.get("PYTHONNOUSERSITE") == "1"
        assert "PYTHONHOME" not in environment and "PYTHONPATH" not in environment
        calls.append(command)

    monkeypatch.setattr(validation, "run", record)
    return calls


def unpublished(monkeypatch: pytest.MonkeyPatch) -> None:
    def absent(root: Path, version: str) -> bool:
        assert root.is_dir() and version
        return False

    monkeypatch.setattr(validation, "release_available", absent)


def test_wheel_selection(wheels: Path) -> None:
    selected = validation.wheel_artifacts(["sefia", "sefia-litellm"], "0.6.0", wheels)
    assert len(selected) == 2
    selected[0].unlink()
    with pytest.raises(ValueError, match="no PyPI fallback"):
        validation.wheel_artifacts(["sefia"], "0.6.0", wheels)
    with pytest.raises(ValueError, match="not candidate version"):
        validation.wheel_artifacts(["sefia-litellm"], "0.6.1", wheels)


def test_duplicate_wheel_fails(wheels: Path) -> None:
    (wheels / "sefia-0.6.0-1-py3-none-any.whl").touch()
    with pytest.raises(ValueError, match="Expected one"):
        validation.wheel_artifacts(["sefia"], "0.6.0", wheels)


def test_candidate_orchestration(
    manifests: Path,
    wheels: Path,
    commands: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/workspace/packages/sefia/src")
    monkeypatch.setenv("PYTHONHOME", "/wrong/python")
    assert validation.validate(manifests, "sefios", "0.6.0", wheels) == "available"
    assert commands[0][:3] == [sys.executable, "-m", "venv"]
    install = commands[1]
    assert "--no-index" in install and "--no-deps" in install
    assert len([arg for arg in install if arg.endswith(".whl")]) == 5
    assert "sefios[all]==0.6.0" in commands[2]
    assert commands[-2][1:] == ["-m", "pip", "check"]
    assert commands[-1][1:3] == ["-m", "pytest"]
    assert commands[-1][3] == str(manifests / "packages/sefios/tests")
    python = Path(commands[-1][0])
    assert not python.parents[2].exists()
    assert all(command[0] == str(python) for command in commands[1:])
    verification = commands[-3]
    assert "strict" == verification[-1]
    assert json.loads(verification[-3]) == {
        name: "0.6.0" for name in dependencies.PACKAGE_ORDER
    }


def test_minimum_candidate_executes(
    manifests: Path,
    wheels: Path,
    commands: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unpublished(monkeypatch)
    assert (
        validation.validate(manifests, "sefios", "0.6.0", wheels, minimum=True)
        == "available"
    )
    install = commands[1]
    assert any(arg.startswith("sefios[all] @ file:") for arg in install)
    assert len([arg for arg in install if arg.endswith(".whl")]) == 5
    assert commands[-2][1:] == ["-m", "pip", "check"]
    assert commands[-1][1:3] == ["-m", "pytest"]
    assert json.loads(commands[-3][-3]) == {
        name: "0.6.0" for name in dependencies.PACKAGE_ORDER
    }


def test_unavailable_and_not_applicable(
    manifests: Path,
    wheels: Path,
    commands: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unpublished(monkeypatch)
    assert (
        validation.validate(manifests, "sefia", "0.6.1", wheels, minimum=True)
        == "not-applicable"
    )
    assert (
        validation.validate(manifests, "sefios", "0.6.1", wheels, minimum=True)
        == "unreleased"
    )
    assert not commands


def test_published_minimum_takes_priority(
    manifests: Path,
    wheels: Path,
    commands: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def published(root: Path, version: str) -> bool:
        assert root.is_dir() and version
        return True

    monkeypatch.setattr(validation, "release_available", published)
    assert (
        validation.validate(manifests, "sefios", "0.6.0", wheels, minimum=True)
        == "available"
    )
    assert len([arg for arg in commands[1] if arg.endswith(".whl")]) == 1


def test_mixed_minima(
    manifests: Path,
    wheels: Path,
    commands: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = manifests / "packages/sefios/pyproject.toml"
    path.write_text(path.read_text().replace("sefia>=0.6.0", "sefia>=0.6.1"))
    for wheel in wheels.iterdir():
        wheel.rename(wheel.with_name(wheel.name.replace("0.6.0", "0.6.1")))

    def published(root: Path, version: str) -> bool:
        assert root.is_dir()
        return version == "0.6.0"

    monkeypatch.setattr(validation, "release_available", published)
    assert (
        validation.validate(manifests, "sefios", "0.6.1", wheels, minimum=True)
        == "available"
    )
    assert len([arg for arg in commands[1] if arg.endswith(".whl")]) == 2
    expected = json.loads(commands[-3][-3])
    assert expected["sefia"] == "0.6.1"
    assert expected["sefia-litellm"] == "0.6.0"


@pytest.mark.parametrize("stage", ["install", "verify", "check", "tests"])
@pytest.mark.parametrize(
    "mode", ["candidate", "minimum-candidate", "minimum-published"]
)
def test_failure_propagation_and_cleanup(
    manifests: Path,
    wheels: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    mode: str,
) -> None:
    def published(root: Path, version: str) -> bool:
        assert root.is_dir() and version
        return mode == "minimum-published"

    monkeypatch.setattr(validation, "release_available", published)
    environments: list[Path] = []

    def fail(command: list[str], root: Path, environment: dict[str, str]) -> None:
        assert root.is_dir() and environment
        if command[1:3] == ["-m", "venv"]:
            environments.append(Path(command[3]).parent)
        current = (
            "install"
            if command[1:4] == ["-m", "pip", "install"]
            else "verify"
            if command[1] == "-c"
            else "check"
            if command[1:4] == ["-m", "pip", "check"]
            else "tests"
            if command[1:3] == ["-m", "pytest"]
            else "venv"
        )
        if current == stage:
            raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(validation, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as error:
        validation.validate(
            manifests, "sefia-litellm", "0.6.0", wheels, minimum=mode != "candidate"
        )
    assert error.value.returncode == 7
    assert environments and not environments[0].exists()


def test_tag_lookup_errors(manifests: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert kwargs["cwd"] == manifests
        return subprocess.CompletedProcess(command, 128)

    monkeypatch.setattr(subprocess, "run", broken)
    with pytest.raises(ValueError, match="Cannot inspect"):
        validation.release_available(manifests, "0.6.0")


def test_real_verifier_rejects_editable_workspace_sources() -> None:
    from importlib import metadata

    root = Path(__file__).resolve().parents[3]
    expected = {name: metadata.version(name) for name in dependencies.PACKAGE_ORDER}
    with pytest.raises(subprocess.CalledProcessError):
        validation.verify_installation(
            root,
            Path(sys.executable),
            expected,
            strict=True,
            environment=validation.clean_environment(),
        )


def test_build_only_runtime_catalog(
    manifests: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "dist"
    built: list[str] = []

    def build(command: list[str], root: Path, environment: dict[str, str]) -> None:
        assert (
            root == manifests
            and environment["SETUPTOOLS_SCM_PRETEND_VERSION"] == "0.6.0"
        )
        path = command[-1]
        name = next(
            name
            for name, package in dependencies.PACKAGES.items()
            if str(root / package["path"]) == path
        )
        built.append(name)
        (output / f"{name.replace('-', '_')}-0.6.0-py3-none-any.whl").touch()

    monkeypatch.setattr(validation, "run", build)
    validation.build_wheels(manifests, "0.6.0", output)
    assert built == list(dependencies.PACKAGE_ORDER)
    with pytest.raises(ValueError, match="must be empty"):
        validation.build_wheels(manifests, "0.6.0", output)


def test_real_isolated_wheel_verification(manifests: Path, tmp_path: Path) -> None:
    import base64
    import hashlib
    import zipfile

    files = {
        "sefia/__init__.py": 'VALUE = "installed wheel"\n',
        "sefia-0.6.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: sefia\nVersion: 0.6.0\n",
        "sefia-0.6.0.dist-info/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    records: list[str] = []
    for name, content in files.items():
        data = content.encode()
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        )
        records.append(f"{name},sha256={digest},{len(data)}\n")
    record = "sefia-0.6.0.dist-info/RECORD"
    files[record] = "".join(records) + f"{record},,\n"
    wheel = tmp_path / "sefia-0.6.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    environment = validation.clean_environment()
    venv = tmp_path / "isolated"
    validation.run([sys.executable, "-m", "venv", str(venv)], manifests, environment)
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    validation.run(
        [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
        manifests,
        environment,
    )
    validation.verify_installation(
        manifests, python, {"sefia": "0.6.0"}, strict=True, environment=environment
    )
    validation.run([str(python), "-m", "pip", "check"], manifests, environment)
