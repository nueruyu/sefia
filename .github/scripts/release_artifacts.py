from __future__ import annotations

import argparse
import json
import re
import tarfile
import zipfile
from email.parser import BytesParser
from importlib import metadata
from pathlib import Path

EXPECTED_DISTRIBUTIONS = (
    "sefia",
    "sefia-litellm",
    "sefia-typer",
    "sefia-fastapi",
    "sefios",
)


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


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


def _discover(
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

    expected = set(EXPECTED_DISTRIBUTIONS)
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
    wheels, sdists = _discover(artifact_dir)

    for name, wheel in wheels.items():
        _, artifact_version = _wheel_metadata(wheel)
        if artifact_version != version:
            raise ValueError(
                f"{wheel.name}: expected version {version}, got {artifact_version}."
            )

    for name, sdist in sdists.items():
        _, artifact_version = _sdist_metadata(sdist)
        if artifact_version != version:
            raise ValueError(
                f"{sdist.name}: expected version {version}, got {artifact_version}."
            )

    return wheels


def write_constraints(artifact_dir: Path, version: str, output: Path) -> None:
    wheels = verify_artifacts(artifact_dir, version)
    lines = [
        f"{name} @ {wheels[name].resolve().as_uri()}"
        for name in EXPECTED_DISTRIBUTIONS
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_installed(
    artifact_dir: Path,
    version: str,
    required_distribution: str,
) -> None:
    wheels = verify_artifacts(artifact_dir, version)
    installed_internal: set[str] = set()

    for name in EXPECTED_DISTRIBUTIONS:
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

    normalized_required = _normalize_name(required_distribution)
    if normalized_required not in installed_internal:
        raise ValueError(
            f"Required distribution {normalized_required} was not installed."
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact-dir", type=Path, required=True)
    verify.add_argument("--version", required=True)

    constraints = subparsers.add_parser("constraints")
    constraints.add_argument("--artifact-dir", type=Path, required=True)
    constraints.add_argument("--version", required=True)
    constraints.add_argument("--output", type=Path, required=True)

    installed = subparsers.add_parser("verify-installed")
    installed.add_argument("--artifact-dir", type=Path, required=True)
    installed.add_argument("--version", required=True)
    installed.add_argument("--distribution", required=True)

    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "verify":
        verify_artifacts(args.artifact_dir, args.version)
    elif args.command == "constraints":
        write_constraints(args.artifact_dir, args.version, args.output)
    elif args.command == "verify-installed":
        verify_installed(args.artifact_dir, args.version, args.distribution)
    else:
        raise AssertionError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
