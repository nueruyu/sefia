from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TypedDict, cast

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


class Package(TypedDict):
    path: str
    requirement: str
    tests: str


Project = TypedDict(
    "Project",
    {
        "name": str,
        "dependencies": list[str],
        "optional-dependencies": dict[str, list[str]],
    },
    total=False,
)


PACKAGES: dict[str, Package] = {
    "sefia": {
        "path": "packages/sefia",
        "requirement": "sefia[testing]",
        "tests": "packages/sefia/tests",
    },
    "sefia-litellm": {
        "path": "packages/sefia_litellm",
        "requirement": "sefia-litellm",
        "tests": "packages/sefia_litellm/tests",
    },
    "sefia-typer": {
        "path": "packages/sefia_typer",
        "requirement": "sefia-typer",
        "tests": "packages/sefia_typer/tests",
    },
    "sefia-fastapi": {
        "path": "packages/sefia_fastapi",
        "requirement": "sefia-fastapi",
        "tests": "packages/sefia_fastapi/tests",
    },
    "sefios": {
        "path": "packages/sefios",
        "requirement": "sefios[all]",
        "tests": "packages/sefios/tests",
    },
}

PACKAGE_ORDER = tuple(PACKAGES)
INTERNAL_DISTRIBUTIONS = frozenset(PACKAGES)


def _project(root: Path, distribution: str) -> Project:
    path = root / PACKAGES[distribution]["path"] / "pyproject.toml"
    return cast(Project, tomllib.loads(path.read_text(encoding="utf-8"))["project"])


def internal_requirements(
    root: Path, distribution: str, extras: set[str], *, conservative: bool = False
) -> list[Requirement]:
    project = _project(root, distribution)
    optional = {
        canonicalize_name(name): requirements
        for name, requirements in project.get("optional-dependencies", {}).items()
    }
    active = set(optional) if conservative else {canonicalize_name(e) for e in extras}
    while True:
        previous = set(active)
        raw = list(project.get("dependencies", []))
        for extra in sorted(active):
            if extra not in optional:
                raise ValueError(f"{distribution} declares no extra {extra!r}")
            raw.extend(optional[extra])
        result: list[Requirement] = []
        for text in raw:
            requirement = Requirement(text)
            if not conservative and requirement.marker is not None:
                if not any(
                    requirement.marker.evaluate({"extra": e}) for e in active | {""}
                ):
                    continue
            name = canonicalize_name(requirement.name)
            if name not in INTERNAL_DISTRIBUTIONS:
                continue
            if requirement.url:
                raise ValueError(f"Unsupported internal URL requirement: {requirement}")
            if name == distribution:
                if requirement.specifier:
                    raise ValueError(
                        f"Unsupported constrained self-reference: {requirement}"
                    )
                active.update(canonicalize_name(e) for e in requirement.extras)
            else:
                result.append(requirement)
        if previous == active:
            return result


def selected_extras(distribution: str) -> set[str]:
    return set(Requirement(PACKAGES[distribution]["requirement"]).extras)


def effective_graph(root: Path, targets: list[str]) -> dict[str, list[Requirement]]:
    extras: dict[str, set[str]] = {name: set() for name in PACKAGE_ORDER}
    selected = set(targets)
    for name in targets:
        extras[name].update(selected_extras(name))
    while True:
        previous = (set(selected), {n: set(e) for n, e in extras.items()})
        for name in PACKAGE_ORDER:
            if name not in selected:
                continue
            for requirement in internal_requirements(root, name, extras[name]):
                dependency = canonicalize_name(requirement.name)
                selected.add(dependency)
                extras[dependency].update(requirement.extras)
        if previous == (selected, extras):
            return {
                name: internal_requirements(root, name, extras[name])
                for name in PACKAGE_ORDER
                if name in selected
            }


def candidate_closure(root: Path, distribution: str) -> list[str]:
    graph = effective_graph(root, [distribution])
    ordered: list[str] = []
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError(f"Internal dependency cycle at {name}")
        if name in ordered:
            return
        visiting.add(name)
        for requirement in graph[name]:
            visit(canonicalize_name(requirement.name))
        visiting.remove(name)
        ordered.append(name)

    for name in PACKAGE_ORDER:
        if name in graph:
            visit(name)
    return ordered


def candidate_version(root: Path, explicit: str | None = None) -> str:
    requirements = [
        (name, requirement)
        for name, dependencies in effective_graph(root, list(PACKAGE_ORDER)).items()
        for requirement in dependencies
    ]
    if explicit is not None:
        candidate = Version(explicit)
        if str(candidate) != explicit:
            raise ValueError(
                f"Use canonical PEP 440 version {candidate}, got {explicit!r}"
            )
    else:
        minimums = [
            Version(spec.version)
            for _, requirement in requirements
            for spec in requirement.specifier
            if spec.operator in {">=", "==", "~="} and "*" not in spec.version
        ]
        candidate = max(minimums, default=Version("0.0.0"))
    for owner, requirement in requirements:
        if candidate not in requirement.specifier:
            raise ValueError(
                f"Candidate {candidate} violates {owner} dependency {requirement}; "
                "update the intended version or deliberately revise package declarations."
            )
    return str(candidate)


def minimum_bounds(root: Path, distribution: str) -> dict[str, str]:
    grouped: dict[str, list[Requirement]] = {}
    for requirement in internal_requirements(
        root, distribution, selected_extras(distribution)
    ):
        grouped.setdefault(canonicalize_name(requirement.name), []).append(requirement)
    bounds: dict[str, str] = {}
    for name, requirements in grouped.items():
        lower = [
            Version(spec.version)
            for r in requirements
            for spec in r.specifier
            if spec.operator in {">=", "==", "~="} and "*" not in spec.version
        ]
        if not lower:
            raise ValueError(
                f"{distribution} dependency {name} has no exact lower bound"
            )
        minimum = max(lower)
        if any(minimum not in r.specifier for r in requirements):
            raise ValueError(
                f"{distribution} has conflicting minimum constraints for {name}"
            )
        bounds[name] = str(minimum)
    return bounds


def repository_root(root: Path) -> Path:
    root = root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError(
            f"Expected a Sefia repository root containing pyproject.toml: {root}"
        )
    for name, package in PACKAGES.items():
        manifest = root / package["path"] / "pyproject.toml"
        if not manifest.is_file():
            raise ValueError(f"Missing {name} manifest at {manifest}")
        actual = _project(root, name).get("name", "")
        if canonicalize_name(actual) != name:
            raise ValueError(f"Expected {name} project at {manifest}, found {actual!r}")
    return root
