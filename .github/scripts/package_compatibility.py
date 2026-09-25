from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from pathlib import Path

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

AFFECTED = {
    "sefia": set(PACKAGE_ORDER),
    "sefia-litellm": {"sefia-litellm", "sefios"},
    "sefia-typer": {"sefia-typer", "sefios"},
    "sefia-fastapi": {"sefia-fastapi", "sefios"},
    "sefios": {"sefios"},
}

CANDIDATE_GRAPH = {
    "sefia": ("sefia",),
    "sefia-litellm": ("sefia", "sefia-litellm"),
    "sefia-typer": ("sefia", "sefia-typer"),
    "sefia-fastapi": ("sefia", "sefia-fastapi"),
    "sefios": PACKAGE_ORDER,
}

GLOBAL_COMPATIBILITY_PATHS = {
    ".github/scripts/package_compatibility.py",
    ".github/workflows/package-compatibility.yml",
    "pyproject.toml",
}


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _version_tuple(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if not 2 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
        raise ValueError(
            "Internal package bounds must use numeric major.minor[.patch] versions; "
            f"got {version!r}."
        )
    values = [int(part) for part in parts]
    while len(values) < 3:
        values.append(0)
    return values[0], values[1], values[2]


def _dependency_strings(distribution: str) -> list[str]:
    path = ROOT / PACKAGES[distribution]["path"] / "pyproject.toml"
    project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    dependencies = list(project.get("dependencies", []))
    for requirements in project.get("optional-dependencies", {}).values():
        dependencies.extend(requirements)
    return dependencies


def _internal_bounds(
    distribution: str,
) -> dict[str, tuple[str, str | None]]:
    bounds: dict[str, tuple[str, str | None]] = {}

    for requirement in _dependency_strings(distribution):
        raw = requirement.split(";", 1)[0].strip()
        match = re.match(
            r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^\]]+\])?(?P<spec>.*)$",
            raw,
        )
        if match is None:
            continue

        name = _normalize_name(match.group("name"))
        if name not in INTERNAL_DISTRIBUTIONS or name == distribution:
            continue

        spec = match.group("spec")
        minimum_match = re.search(r"(?:^|,)\s*>=\s*([^,\s]+)", spec)
        maximum_match = re.search(r"(?:^|,)\s*<\s*([^,\s]+)", spec)
        if minimum_match is None:
            raise ValueError(
                f"{distribution} internal dependency {requirement!r} has no >= bound."
            )

        minimum = minimum_match.group(1)
        maximum = maximum_match.group(1) if maximum_match is not None else None
        previous = bounds.get(name)
        if previous is None or _version_tuple(minimum) > _version_tuple(previous[0]):
            bounds[name] = minimum, maximum

    return bounds


def candidate_version() -> str:
    all_bounds = [
        bound
        for distribution in PACKAGE_ORDER
        for bound in _internal_bounds(distribution).values()
    ]
    if not all_bounds:
        raise ValueError("No internal package bounds were found.")

    minimums = [_version_tuple(minimum) for minimum, _ in all_bounds]
    highest = max(minimums)
    patches = [
        patch for major, minor, patch in minimums if (major, minor) == highest[:2]
    ]
    candidate = highest[0], highest[1], max(patches) + 1000

    for _, maximum in all_bounds:
        if maximum is not None and candidate >= _version_tuple(maximum):
            raise ValueError(
                f"Synthetic candidate {candidate} does not satisfy upper bound "
                f"{maximum}."
            )

    return ".".join(str(part) for part in candidate)


def affected_packages(base: str, head: str) -> list[str]:
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", base, head],
        cwd=ROOT,
        text=True,
    ).splitlines()

    affected: set[str] = set()
    for path in changed:
        if path in GLOBAL_COMPATIBILITY_PATHS:
            return list(PACKAGE_ORDER)

        for distribution, package in PACKAGES.items():
            if path.startswith(f"{package['path']}/"):
                affected.update(AFFECTED[distribution])
                break

    return [name for name in PACKAGE_ORDER if name in affected]


def matrix(distributions: list[str]) -> dict[str, list[dict[str, str]]]:
    return {
        "include": [
            {
                "distribution": distribution,
                "requirement": PACKAGES[distribution]["requirement"],
                "tests": PACKAGES[distribution]["tests"],
                "candidate-distributions": " ".join(CANDIDATE_GRAPH[distribution]),
            }
            for distribution in distributions
        ]
    }


def write_minimum_constraints(distribution: str, output: Path) -> None:
    bounds = _internal_bounds(distribution)
    lines = [f"{name}=={minimum}" for name, (minimum, _) in sorted(bounds.items())]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def check_minimum_released(distribution: str) -> bool:
    bounds = _internal_bounds(distribution)
    if not bounds:
        print(f"{distribution} has no internal lower bounds to test.")
        return False

    versions = {minimum for minimum, _ in bounds.values()}
    missing = [
        version
        for version in sorted(versions)
        if subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/tags/v{version}"],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    ]

    if missing:
        tags = ", ".join(f"v{version}" for version in missing)
        print(f"Minimum compatibility is waiting for release tag(s): {tags}")
        return False

    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("candidate-version")

    selected = subparsers.add_parser("matrix")
    selected.add_argument("--base")
    selected.add_argument("--head")
    selected.add_argument("--all", action="store_true")

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
        print(candidate_version())
    elif args.command == "matrix":
        print(json.dumps(matrix(_selected_packages(args)), separators=(",", ":")))
    elif args.command == "minimum-constraints":
        write_minimum_constraints(args.distribution, args.output)
    elif args.command == "check-minimum-released":
        raise SystemExit(0 if check_minimum_released(args.distribution) else 2)
    else:
        raise AssertionError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
