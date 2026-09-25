from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PACKAGES = {
    "sefia": {
        "path": "packages/sefia",
        "requirement": "sefia[testing]",
        "module": "sefia",
        "tests": "packages/sefia/tests",
    },
    "sefia-litellm": {
        "path": "packages/sefia_litellm",
        "requirement": "sefia-litellm",
        "module": "sefia_litellm",
        "tests": "packages/sefia_litellm/tests",
    },
    "sefia-typer": {
        "path": "packages/sefia_typer",
        "requirement": "sefia-typer",
        "module": "sefia_typer",
        "tests": "packages/sefia_typer/tests",
    },
    "sefia-fastapi": {
        "path": "packages/sefia_fastapi",
        "requirement": "sefia-fastapi",
        "module": "sefia_fastapi",
        "tests": "packages/sefia_fastapi/tests",
    },
    "sefios": {
        "path": "packages/sefios",
        "requirement": "sefios[all]",
        "module": "sefios",
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
    padded = [int(part) for part in parts]
    while len(padded) < 3:
        padded.append(0)
    return tuple(padded)  # type: ignore[return-value]


def _dependency_strings(distribution: str) -> list[str]:
    path = ROOT / PACKAGES[distribution]["path"] / "pyproject.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data["project"]
    dependencies = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    for requirements in optional.values():
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
            bounds[name] = (minimum, maximum)

    return bounds


def candidate_version() -> str:
    all_bounds: list[tuple[str, str | None]] = []
    for distribution in PACKAGE_ORDER:
        all_bounds.extend(_internal_bounds(distribution).values())

    if not all_bounds:
        raise ValueError("No internal package bounds were found.")

    minimums = [_version_tuple(minimum) for minimum, _ in all_bounds]
    highest = max(minimums)
    same_line_patches = [
        patch
        for major, minor, patch in minimums
        if (major, minor) == highest[:2]
    ]
    candidate = (highest[0], highest[1], max(same_line_patches) + 1000)

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
            prefix = f"{package['path']}/"
            if path.startswith(prefix):
                affected.update(AFFECTED[distribution])
                break

    return [name for name in PACKAGE_ORDER if name in affected]


def matrix(distributions: list[str]) -> dict[str, list[dict[str, str]]]:
    include = []
    for distribution in distributions:
        package = PACKAGES[distribution]
        include.append(
            {
                "distribution": distribution,
                "requirement": package["requirement"],
                "module": package["module"],
                "tests": package["tests"],
            }
        )
    return {"include": include}


def minimum_matrix(distributions: list[str]) -> dict[str, list[dict[str, str]]]:
    minimum_distributions = [
        distribution
        for distribution in distributions
        if _internal_bounds(distribution)
    ]
    return matrix(minimum_distributions)


def _metadata_fields(data: bytes) -> tuple[str, str]:
    message = BytesParser().parsebytes(data)
    name = message["Name"]
    version = message["Version"]
    if not name or not version:
        raise ValueError("Artifact metadata must include Name and Version.")
    return _normalize_name(name), version


def _wheel_metadata(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError(
                f"{path.name}: expected one wheel METADATA file, "
                f"found {len(metadata_files)}."
            )
        return _metadata_fields(archive.read(metadata_files[0]))


def _sdist_metadata(path: Path) -> tuple[str, str]:
    with tarfile.open(path, "r:gz") as archive:
        metadata_files = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith("/PKG-INFO")
        ]
        if len(metadata_files) != 1:
            raise ValueError(
                f"{path.name}: expected one sdist PKG-INFO file, "
                f"found {len(metadata_files)}."
            )
        extracted = archive.extractfile(metadata_files[0])
        if extracted is None:
            raise ValueError(f"{path.name}: could not read PKG-INFO.")
        return _metadata_fields(extracted.read())


def _discover_artifacts(
    artifact_dir: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    wheels: dict[str, Path] = {}
    sdists: dict[str, Path] = {}

    for path in artifact_dir.glob("*.whl"):
        name, _ = _wheel_metadata(path)
        if name in wheels:
            raise ValueError(f"Duplicate wheel for {name}: {path.name}")
        wheels[name] = path

    for path in artifact_dir.glob("*.tar.gz"):
        name, _ = _sdist_metadata(path)
        if name in sdists:
            raise ValueError(f"Duplicate sdist for {name}: {path.name}")
        sdists[name] = path

    expected = set(INTERNAL_DISTRIBUTIONS)
    if set(wheels) != expected:
        raise ValueError(
            f"Wheel set mismatch: expected {sorted(expected)}, got {sorted(wheels)}."
        )
    if set(sdists) != expected:
        raise ValueError(
            f"Sdist set mismatch: expected {sorted(expected)}, got {sorted(sdists)}."
        )

    return wheels, sdists


def verify_artifacts(artifact_dir: Path, version: str) -> dict[str, Path]:
    wheels, sdists = _discover_artifacts(artifact_dir)

    for wheel in wheels.values():
        _, artifact_version = _wheel_metadata(wheel)
        if artifact_version != version:
            raise ValueError(
                f"{wheel.name}: expected version {version}, got {artifact_version}."
            )

    for sdist in sdists.values():
        _, artifact_version = _sdist_metadata(sdist)
        if artifact_version != version:
            raise ValueError(
                f"{sdist.name}: expected version {version}, got {artifact_version}."
            )

    return wheels


def write_candidate_constraints(
    artifact_dir: Path,
    version: str,
    output: Path,
) -> None:
    wheels = verify_artifacts(artifact_dir, version)
    lines = [
        f"{name} @ {wheels[name].resolve().as_uri()}" for name in PACKAGE_ORDER
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_minimum_constraints(distribution: str, output: Path) -> None:
    bounds = _internal_bounds(distribution)
    lines = [f"{name}=={minimum}" for name, (minimum, _) in sorted(bounds.items())]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_minimum_released(distribution: str) -> bool:
    versions = {minimum for minimum, _ in _internal_bounds(distribution).values()}
    missing = []

    for version in sorted(versions):
        result = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/tags/v{version}"],
            cwd=ROOT,
            check=False,
        )
        if result.returncode != 0:
            missing.append(version)

    if missing:
        print(
            "Minimum compatibility is not runnable yet; "
            f"unreleased lower-bound tag(s): {', '.join('v' + v for v in missing)}"
        )
        return False

    return True


def wheel_requirement(
    artifact_dir: Path,
    version: str,
    distribution: str,
) -> str:
    wheels = verify_artifacts(artifact_dir, version)
    requirement = PACKAGES[distribution]["requirement"]
    return f"{requirement} @ {wheels[distribution].resolve().as_uri()}"


def verify_candidate_installed(
    artifact_dir: Path,
    version: str,
    required_distribution: str,
) -> None:
    wheels = verify_artifacts(artifact_dir, version)
    installed_internal: set[str] = set()

    for name in PACKAGE_ORDER:
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue

        installed_internal.add(name)
        if distribution.version != version:
            raise ValueError(
                f"Installed {name} has version {distribution.version}, "
                f"expected {version}."
            )

        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text is None:
            raise ValueError(f"Installed {name} is not recorded as a direct artifact.")

        direct_url = json.loads(direct_url_text).get("url")
        expected_url = wheels[name].resolve().as_uri()
        if direct_url != expected_url:
            raise ValueError(
                f"Installed {name} came from {direct_url!r}, "
                f"expected candidate wheel {expected_url!r}."
            )

    if required_distribution not in installed_internal:
        raise ValueError(
            f"Required distribution {required_distribution} was not installed."
        )


def verify_minimum_installed(distribution: str, constraints: Path) -> None:
    expected = {}
    for line in constraints.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        name, version = line.split("==", 1)
        expected[name] = version

    for name, version in expected.items():
        actual = metadata.version(name)
        if actual != version:
            raise ValueError(f"Installed {name} is {actual}, expected {version}.")

    target = metadata.distribution(distribution)
    direct_url_text = target.read_text("direct_url.json")
    if direct_url_text is None:
        raise ValueError(
            f"Current {distribution} was not installed from the candidate wheel."
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidate = subparsers.add_parser("candidate-version")

    affected = subparsers.add_parser("matrix")
    affected.add_argument("--base")
    affected.add_argument("--head")
    affected.add_argument("--all", action="store_true")

    minimum = subparsers.add_parser("minimum-matrix")
    minimum.add_argument("--base")
    minimum.add_argument("--head")
    minimum.add_argument("--all", action="store_true")

    verify = subparsers.add_parser("verify-artifacts")
    verify.add_argument("--artifact-dir", type=Path, required=True)
    verify.add_argument("--version", required=True)

    constraints = subparsers.add_parser("candidate-constraints")
    constraints.add_argument("--artifact-dir", type=Path, required=True)
    constraints.add_argument("--version", required=True)
    constraints.add_argument("--output", type=Path, required=True)

    minimum_constraints = subparsers.add_parser("minimum-constraints")
    minimum_constraints.add_argument("--distribution", required=True)
    minimum_constraints.add_argument("--output", type=Path, required=True)

    released = subparsers.add_parser("check-minimum-released")
    released.add_argument("--distribution", required=True)

    requirement = subparsers.add_parser("wheel-requirement")
    requirement.add_argument("--artifact-dir", type=Path, required=True)
    requirement.add_argument("--version", required=True)
    requirement.add_argument("--distribution", required=True)

    candidate_installed = subparsers.add_parser("verify-candidate-installed")
    candidate_installed.add_argument("--artifact-dir", type=Path, required=True)
    candidate_installed.add_argument("--version", required=True)
    candidate_installed.add_argument("--distribution", required=True)

    minimum_installed = subparsers.add_parser("verify-minimum-installed")
    minimum_installed.add_argument("--distribution", required=True)
    minimum_installed.add_argument("--constraints", type=Path, required=True)

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
    elif args.command == "minimum-matrix":
        selected = _selected_packages(args)
        print(json.dumps(minimum_matrix(selected), separators=(",", ":")))
    elif args.command == "verify-artifacts":
        verify_artifacts(args.artifact_dir, args.version)
    elif args.command == "candidate-constraints":
        write_candidate_constraints(args.artifact_dir, args.version, args.output)
    elif args.command == "minimum-constraints":
        write_minimum_constraints(args.distribution, args.output)
    elif args.command == "check-minimum-released":
        raise SystemExit(0 if check_minimum_released(args.distribution) else 2)
    elif args.command == "wheel-requirement":
        print(
            wheel_requirement(
                args.artifact_dir,
                args.version,
                args.distribution,
            )
        )
    elif args.command == "verify-candidate-installed":
        verify_candidate_installed(
            args.artifact_dir,
            args.version,
            args.distribution,
        )
    elif args.command == "verify-minimum-installed":
        verify_minimum_installed(args.distribution, args.constraints)
    else:
        raise AssertionError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
