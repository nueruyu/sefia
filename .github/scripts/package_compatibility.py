from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]

PACKAGES = {
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

GLOBAL_COMPATIBILITY_PATHS = {
    ".github/scripts/package_compatibility.py",
    ".github/workflows/package-compatibility.yml",
    "pyproject.toml",
}


def _project(distribution: str) -> dict:
    path = ROOT / PACKAGES[distribution]["path"] / "pyproject.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def internal_requirements(
    distribution: str, extras: set[str], *, conservative: bool = False
) -> list[Requirement]:
    project = _project(distribution)
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


def effective_graph(targets: list[str]) -> dict[str, list[Requirement]]:
    extras = {name: set() for name in PACKAGE_ORDER}
    selected = set(targets)
    for name in targets:
        extras[name].update(selected_extras(name))
    while True:
        previous = (set(selected), {n: set(e) for n, e in extras.items()})
        for name in PACKAGE_ORDER:
            if name not in selected:
                continue
            for requirement in internal_requirements(name, extras[name]):
                dependency = canonicalize_name(requirement.name)
                selected.add(dependency)
                extras[dependency].update(requirement.extras)
        if previous == (selected, extras):
            return {
                name: internal_requirements(name, extras[name])
                for name in PACKAGE_ORDER
                if name in selected
            }


def candidate_closure(distribution: str) -> list[str]:
    graph = effective_graph([distribution])
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


def candidate_version(explicit: str | None = None) -> str:
    requirements = [
        (name, requirement)
        for name, dependencies in effective_graph(list(PACKAGE_ORDER)).items()
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
        highest = max(minimums, default=Version("0.0.0"))
        release = highest.release + (0,) * (3 - len(highest.release))
        candidate = Version(f"{release[0]}.{release[1]}.{release[2] + 1000}")
    for owner, requirement in requirements:
        if candidate not in requirement.specifier:
            raise ValueError(
                f"Candidate {candidate} violates {owner} dependency {requirement}; "
                "update the intended version or deliberately revise package declarations."
            )
    source = (
        "explicit manual candidate"
        if explicit is not None
        else "synthetic PR candidate"
    )
    print(f"Candidate version: {candidate} ({source})", file=sys.stderr)
    return str(candidate)


def affected_paths(changed: list[str]) -> list[str]:
    if any(
        path in GLOBAL_COMPATIBILITY_PATHS or path.startswith("tests/tooling/")
        for path in changed
    ):
        return list(PACKAGE_ORDER)
    affected = {
        name
        for name, package in PACKAGES.items()
        if any(path.startswith(f"{package['path']}/") for path in changed)
    }
    while True:
        previous = set(affected)
        for name in PACKAGE_ORDER:
            dependencies = internal_requirements(name, set(), conservative=True)
            if any(canonicalize_name(r.name) in affected for r in dependencies):
                affected.add(name)
        if previous == affected:
            return [name for name in PACKAGE_ORDER if name in affected]


def affected_packages(base: str, head: str) -> list[str]:
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", f"{base}...{head}"], cwd=ROOT, text=True
    ).splitlines()
    return affected_paths(changed)


def matrix(distributions: list[str]) -> dict[str, list[dict[str, str]]]:
    return {
        "include": [
            {
                "distribution": distribution,
                "requirement": PACKAGES[distribution]["requirement"],
                "tests": PACKAGES[distribution]["tests"],
                "candidate-distributions": " ".join(candidate_closure(distribution)),
            }
            for distribution in distributions
        ]
    }


def candidate_artifacts(distribution: str, version: str, directory: Path) -> list[Path]:
    wheels: dict[str, list[Path]] = {}
    for path in sorted(directory.glob("*.whl")):
        name, wheel_version, _, _ = parse_wheel_filename(path.name)
        if name in INTERNAL_DISTRIBUTIONS:
            if wheel_version != Version(version):
                raise ValueError(f"{path.name} is not candidate version {version}")
            wheels.setdefault(name, []).append(path)
    selected = candidate_closure(distribution)
    for name in selected:
        if len(wheels.get(name, [])) != 1:
            raise ValueError(
                f"Expected one candidate wheel for {name}; no PyPI fallback permitted"
            )
    return [wheels[name][0] for name in selected]


def write_candidate_constraints(version: str, output: Path) -> None:
    output.write_text(
        "".join(f"{name}=={version}\n" for name in PACKAGE_ORDER), encoding="utf-8"
    )


def minimum_bounds(distribution: str) -> dict[str, str]:
    grouped: dict[str, list[Requirement]] = {}
    for requirement in internal_requirements(
        distribution, selected_extras(distribution)
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


def write_minimum_constraints(distribution: str, output: Path) -> None:
    lines = [
        f"{name}=={version}"
        for name, version in sorted(minimum_bounds(distribution).items())
    ]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def minimum_status(distribution: str) -> str:
    bounds = minimum_bounds(distribution)
    if not bounds:
        print(f"{distribution}: no applicable internal dependencies")
        return "not-applicable"
    missing = []
    for version in sorted(set(bounds.values())):
        status = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/tags/v{version}"],
            cwd=ROOT,
            check=False,
        ).returncode
        if status not in {0, 1}:
            raise ValueError(
                f"Cannot inspect release tag v{version}: git exited {status}"
            )
        if status:
            missing.append(f"v{version}")
    if missing:
        print(f"{distribution}: unreleased minimum; missing tags: {', '.join(missing)}")
        return "unreleased"
    print(f"{distribution}: minimum available; installation and tests required")
    return "available"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    version = subparsers.add_parser("candidate-version")
    version.add_argument("--version")

    selected = subparsers.add_parser("matrix")
    selected.add_argument("--base")
    selected.add_argument("--head")
    selected.add_argument("--all", action="store_true")

    artifacts = subparsers.add_parser("candidate-artifacts")
    artifacts.add_argument("--distribution", required=True, choices=PACKAGE_ORDER)
    artifacts.add_argument("--version", required=True)
    artifacts.add_argument("--directory", type=Path, required=True)
    artifacts.add_argument("--constraints", type=Path, required=True)

    minimum = subparsers.add_parser("minimum-constraints")
    minimum.add_argument("--distribution", required=True)
    minimum.add_argument("--output", type=Path, required=True)

    released = subparsers.add_parser("check-minimum-released")
    released.add_argument("--distribution", required=True)

    return parser


def _selected_packages(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(PACKAGE_ORDER)
    if not args.base or not args.head:
        raise ValueError("--base and --head are required without --all.")
    return affected_packages(args.base, args.head)


def main() -> None:
    args = _parser().parse_args()

    if args.command == "candidate-version":
        print(candidate_version(args.version))
    elif args.command == "matrix":
        print(json.dumps(matrix(_selected_packages(args)), separators=(",", ":")))
    elif args.command == "candidate-artifacts":
        wheels = candidate_artifacts(args.distribution, args.version, args.directory)
        write_candidate_constraints(args.version, args.constraints)
        print("\n".join(str(path) for path in wheels))
    elif args.command == "minimum-constraints":
        write_minimum_constraints(args.distribution, args.output)
    elif args.command == "check-minimum-released":
        status = minimum_status(args.distribution)
        raise SystemExit({"available": 0, "unreleased": 2, "not-applicable": 3}[status])
    else:
        raise AssertionError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
