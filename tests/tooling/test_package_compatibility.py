from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from packaging.version import InvalidVersion, Version

SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github/scripts/package_compatibility.py"
)


@pytest.fixture
def helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("package_compatibility", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def manifests(
    helper: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    for package in helper.PACKAGES.values():
        relative = Path(package["path"]) / "pyproject.toml"
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((helper.ROOT / relative).read_text())
    monkeypatch.setattr(helper, "ROOT", tmp_path)
    return tmp_path


def add_dependency(root: Path, package: str, requirement: str) -> None:
    path = root / "packages" / package / "pyproject.toml"
    text = path.read_text().replace(
        "dependencies = [", f"dependencies = [\n    {json.dumps(requirement)},", 1
    )
    path.write_text(text)


def test_explicit_versions(helper: ModuleType) -> None:
    assert helper.candidate_version("0.6.1") == "0.6.1"
    with pytest.raises(ValueError, match=r"sefia.*<0.7.0"):
        helper.candidate_version("0.7.0")
    with pytest.raises(InvalidVersion):
        helper.candidate_version("not-a-version")


def test_synthetic_version(helper: ModuleType) -> None:
    version = helper.candidate_version()
    assert version == helper.candidate_version()
    for name in helper.PACKAGE_ORDER:
        for requirement in helper.internal_requirements(
            name, helper.selected_extras(name)
        ):
            assert Version(version) in requirement.specifier


def test_impossible_and_conflicting_bounds(helper: ModuleType, manifests: Path) -> None:
    add_dependency(manifests, "sefia_litellm", "sefia>=0.8.0,<0.9.0")
    with pytest.raises(ValueError, match="violates"):
        helper.candidate_version()
    with pytest.raises(ValueError, match="conflicting"):
        helper.minimum_bounds("sefia-litellm")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "packages/sefia/src/change.py",
            ["sefia", "sefia-litellm", "sefia-typer", "sefia-fastapi", "sefios"],
        ),
        ("packages/sefia_litellm/pyproject.toml", ["sefia-litellm", "sefios"]),
        ("packages/sefios/src/change.py", ["sefios"]),
        ("docs/tutorial.md", []),
        (
            "pyproject.toml",
            ["sefia", "sefia-litellm", "sefia-typer", "sefia-fastapi", "sefios"],
        ),
        (
            ".github/scripts/package_compatibility.py",
            ["sefia", "sefia-litellm", "sefia-typer", "sefia-fastapi", "sefios"],
        ),
    ],
)
def test_impact(helper: ModuleType, path: str, expected: list[str]) -> None:
    assert helper.affected_paths([path]) == expected


def test_new_normalized_transitive_edge(helper: ModuleType, manifests: Path) -> None:
    add_dependency(manifests, "sefia_typer", "Sefia_LiteLLM [extra] >=0.6.0, <0.7.0")
    assert helper.affected_paths(["packages/sefia_litellm/pyproject.toml"]) == [
        "sefia-litellm",
        "sefia-typer",
        "sefios",
    ]
    # The incoming extra is validated rather than silently dropped.
    with pytest.raises(ValueError, match="no extra"):
        helper.candidate_closure("sefia-typer")


def test_extra_expansion_and_candidate_closure(helper: ModuleType) -> None:
    assert helper.candidate_closure("sefios") == list(helper.PACKAGE_ORDER)
    assert helper.candidate_closure("sefia-fastapi") == ["sefia", "sefia-fastapi"]
    assert {r.name for r in helper.internal_requirements("sefios", {"all"})} == {
        "sefia",
        "sefia-litellm",
        "sefia-typer",
        "sefia-fastapi",
    }


def test_new_edge_closure_and_cycle(helper: ModuleType, manifests: Path) -> None:
    add_dependency(manifests, "sefia_typer", "sefia_litellm>=0.6.0,<0.7.0")
    assert helper.candidate_closure("sefia-typer") == [
        "sefia",
        "sefia-litellm",
        "sefia-typer",
    ]
    add_dependency(manifests, "sefia", "sefia-typer>=0.6.0,<0.7.0")
    with pytest.raises(ValueError, match="cycle"):
        helper.candidate_closure("sefia-typer")


def test_markers(helper: ModuleType, manifests: Path) -> None:
    add_dependency(manifests, "sefia", 'sefia-typer>=0.6.0; python_version < "2"')
    assert helper.candidate_closure("sefia") == ["sefia"]
    assert "sefia" in helper.affected_paths(["packages/sefia_typer/src/change.py"])


def test_minimum_states(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert helper.minimum_status("sefia") == "not-applicable"
    monkeypatch.setattr(
        helper.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1)
    )
    assert helper.minimum_status("sefios") == "unreleased"
    monkeypatch.setattr(
        helper.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    assert helper.minimum_status("sefios") == "available"
    output = tmp_path / "constraints.txt"
    helper.write_minimum_constraints("sefios", output)
    assert (
        output.read_text()
        == "sefia==0.6.0\nsefia-fastapi==0.6.0\nsefia-litellm==0.6.0\nsefia-typer==0.6.0\n"
    )
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=128),
    )
    with pytest.raises(ValueError, match="Cannot inspect"):
        helper.minimum_status("sefios")


def test_unsupported_requirements(helper: ModuleType, manifests: Path) -> None:
    add_dependency(manifests, "sefia", "sefia-typer @ https://example.com/wheel.whl")
    with pytest.raises(ValueError, match="Unsupported internal URL"):
        helper.candidate_closure("sefia")


def test_merge_base_selection(helper: ModuleType, manifests: Path) -> None:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=manifests, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.email", "fixture@example.com")
    git("config", "user.name", "Fixture")
    git("add", ".")
    git("commit", "-m", "initial")
    git("switch", "-c", "feature")
    (manifests / "README.md").write_text("documentation only")
    git("add", ".")
    git("commit", "-m", "docs")
    git("switch", "main")
    add_dependency(manifests, "sefia", "typing-extensions>=4.6")
    git("add", ".")
    git("commit", "-m", "base-only change")
    assert helper.affected_packages("main", "feature") == []


def test_cli(helper: ModuleType) -> None:
    import sys

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "candidate-version", "--version", "0.6.1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "0.6.1"
    assert "explicit manual" in result.stderr
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "candidate-version", "--version", "0.7.0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "violates" in result.stderr
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "matrix", "--all"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(json.loads(result.stdout)["include"]) == 5


def test_candidate_artifacts_no_pypi_fallback(
    helper: ModuleType, tmp_path: Path
) -> None:
    for name in helper.PACKAGE_ORDER:
        (tmp_path / f"{name.replace('-', '_')}-0.6.1000-py3-none-any.whl").touch()
    wheels = helper.candidate_artifacts("sefios", "0.6.1000", tmp_path)
    assert len(wheels) == 5
    assert len(helper.candidate_artifacts("sefia-litellm", "0.6.1000", tmp_path)) == 2
    constraints = tmp_path / "constraints.txt"
    helper.write_candidate_constraints("0.6.1000", constraints)
    assert constraints.read_text().splitlines() == [
        f"{n}==0.6.1000" for n in helper.PACKAGE_ORDER
    ]
    wheels[0].unlink()
    with pytest.raises(ValueError, match="no PyPI fallback"):
        helper.candidate_artifacts("sefios", "0.6.1000", tmp_path)
    with pytest.raises(ValueError, match="not candidate version"):
        helper.candidate_artifacts("sefia-litellm", "0.6.1", tmp_path)


def test_inherited_extra_version_validation(
    helper: ModuleType, manifests: Path
) -> None:
    path = manifests / "packages/sefia_typer/pyproject.toml"
    path.write_text(
        path.read_text()
        + '\n[project.optional-dependencies]\nfuture = ["sefia>=0.7.0"]\n'
    )
    add_dependency(manifests, "sefia_litellm", "sefia-typer[future]>=0.6.0,<0.7.0")
    with pytest.raises(ValueError, match="sefia-typer dependency sefia>=0.7.0"):
        helper.candidate_version("0.6.1")
