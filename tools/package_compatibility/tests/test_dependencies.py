from pathlib import Path
import json
import subprocess

import pytest
from packaging.version import InvalidVersion, Version
from package_compatibility import dependencies


def add_dependency(root: Path, package: str, requirement: str) -> None:
    path = root / "packages" / package / "pyproject.toml"
    text = path.read_text().replace(
        "dependencies = [", f"dependencies = [\n    {json.dumps(requirement)},", 1
    )
    path.write_text(text)


def test_explicit_versions(manifests: Path) -> None:
    assert dependencies.candidate_version(manifests, "0.6.1") == "0.6.1"
    with pytest.raises(ValueError, match=r"sefia.*<0.7.0"):
        dependencies.candidate_version(manifests, "0.7.0")
    with pytest.raises(InvalidVersion):
        dependencies.candidate_version(manifests, "not-a-version")


def test_synthetic_version(manifests: Path) -> None:
    version = dependencies.candidate_version(
        manifests,
    )
    assert version == dependencies.candidate_version(
        manifests,
    )
    for name in dependencies.PACKAGE_ORDER:
        for requirement in dependencies.internal_requirements(
            manifests, name, dependencies.selected_extras(name)
        ):
            assert Version(version) in requirement.specifier


def test_impossible_and_conflicting_bounds(manifests: Path) -> None:
    add_dependency(manifests, "sefia_litellm", "sefia>=0.8.0,<0.9.0")
    with pytest.raises(ValueError, match="violates"):
        dependencies.candidate_version(
            manifests,
        )
    with pytest.raises(ValueError, match="conflicting"):
        dependencies.minimum_bounds(manifests, "sefia-litellm")


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
        ("uv.lock", list(dependencies.PACKAGE_ORDER)),
        (
            ".github/workflows/package-compatibility.yml",
            list(dependencies.PACKAGE_ORDER),
        ),
        (
            "tools/package_compatibility/pyproject.toml",
            list(dependencies.PACKAGE_ORDER),
        ),
        (
            "tools/package_compatibility/tests/test_cli.py",
            list(dependencies.PACKAGE_ORDER),
        ),
        (
            "pyproject.toml",
            ["sefia", "sefia-litellm", "sefia-typer", "sefia-fastapi", "sefios"],
        ),
        (
            "tools/package_compatibility/src/package_compatibility/validation.py",
            ["sefia", "sefia-litellm", "sefia-typer", "sefia-fastapi", "sefios"],
        ),
    ],
)
def test_impact(manifests: Path, path: str, expected: list[str]) -> None:
    assert dependencies.affected_paths(manifests, [path]) == expected


def test_new_normalized_transitive_edge(manifests: Path) -> None:
    add_dependency(manifests, "sefia_typer", "Sefia_LiteLLM [extra] >=0.6.0, <0.7.0")
    assert dependencies.affected_paths(
        manifests, ["packages/sefia_litellm/pyproject.toml"]
    ) == [
        "sefia-litellm",
        "sefia-typer",
        "sefios",
    ]
    # The incoming extra is validated rather than silently dropped.
    with pytest.raises(ValueError, match="no extra"):
        dependencies.candidate_closure(manifests, "sefia-typer")


def test_extra_expansion_and_candidate_closure(manifests: Path) -> None:
    assert dependencies.candidate_closure(manifests, "sefios") == list(
        dependencies.PACKAGE_ORDER
    )
    assert dependencies.candidate_closure(manifests, "sefia-fastapi") == [
        "sefia",
        "sefia-fastapi",
    ]
    assert {
        r.name for r in dependencies.internal_requirements(manifests, "sefios", {"all"})
    } == {
        "sefia",
        "sefia-litellm",
        "sefia-typer",
        "sefia-fastapi",
    }


def test_new_edge_closure_and_cycle(manifests: Path) -> None:
    add_dependency(manifests, "sefia_typer", "sefia_litellm>=0.6.0,<0.7.0")
    assert dependencies.candidate_closure(manifests, "sefia-typer") == [
        "sefia",
        "sefia-litellm",
        "sefia-typer",
    ]
    add_dependency(manifests, "sefia", "sefia-typer>=0.6.0,<0.7.0")
    with pytest.raises(ValueError, match="cycle"):
        dependencies.candidate_closure(manifests, "sefia-typer")


def test_markers(manifests: Path) -> None:
    add_dependency(manifests, "sefia", 'sefia-typer>=0.6.0; python_version < "2"')
    assert dependencies.candidate_closure(manifests, "sefia") == ["sefia"]
    assert "sefia" in dependencies.affected_paths(
        manifests, ["packages/sefia_typer/src/change.py"]
    )


def test_unsupported_requirements(manifests: Path) -> None:
    add_dependency(manifests, "sefia", "sefia-typer @ https://example.com/wheel.whl")
    with pytest.raises(ValueError, match="Unsupported internal URL"):
        dependencies.candidate_closure(manifests, "sefia")


def test_merge_base_selection(manifests: Path) -> None:
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
    assert dependencies.affected_packages(manifests, "main", "feature") == []


def test_inherited_extra_version_validation(manifests: Path) -> None:
    path = manifests / "packages/sefia_typer/pyproject.toml"
    path.write_text(
        path.read_text()
        + '\n[project.optional-dependencies]\nfuture = ["sefia>=0.7.0"]\n'
    )
    add_dependency(manifests, "sefia_litellm", "sefia-typer[future]>=0.6.0,<0.7.0")
    with pytest.raises(ValueError, match="sefia-typer dependency sefia>=0.7.0"):
        dependencies.candidate_version(manifests, "0.6.1")


def test_fixture_release_range_can_advance(manifests: Path) -> None:
    for path in manifests.glob("packages/*/pyproject.toml"):
        path.write_text(path.read_text().replace(">=0.6.0,<0.7.0", ">=0.7.0,<0.8.0"))
    assert dependencies.candidate_version(manifests, "0.7.1") == "0.7.1"
    with pytest.raises(ValueError, match="violates"):
        dependencies.candidate_version(manifests, "0.8.0")
    assert dependencies.minimum_bounds(manifests, "sefia-litellm") == {"sefia": "0.7.0"}
    assert (
        Version("0.7.0")
        <= Version(
            dependencies.candidate_version(
                manifests,
            )
        )
        < Version("0.8.0")
    )


def test_checkout_metadata_smoke() -> None:
    manifests = Path(__file__).resolve().parents[3]
    version = Version(
        dependencies.candidate_version(
            manifests,
        )
    )
    graph = dependencies.effective_graph(manifests, list(dependencies.PACKAGE_ORDER))
    assert set(graph) == set(dependencies.PACKAGE_ORDER)
    for requirements in graph.values():
        assert all(version in requirement.specifier for requirement in requirements)
    for name in dependencies.PACKAGE_ORDER:
        assert set(dependencies.candidate_closure(manifests, name)) <= set(
            dependencies.PACKAGE_ORDER
        )
