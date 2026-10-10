from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from packaging.utils import parse_wheel_filename
from packaging.version import Version

from .dependencies import (
    INTERNAL_DISTRIBUTIONS,
    PACKAGES,
    PACKAGE_ORDER,
    candidate_closure,
    candidate_version,
    minimum_bounds,
)

MinimumStatus = Literal["available", "unreleased", "not-applicable"]
TEST_REQUIREMENTS = ("pytest>=8.0", "pytest-asyncio>=0.23", "pytest-mock>=3.12.0")


def clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def run(command: list[str], root: Path, environment: dict[str, str]) -> None:
    print("Running: " + " ".join(command), file=sys.stderr, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)


def wheel_artifacts(names: list[str], version: str, directory: Path) -> list[Path]:
    wheels: dict[str, list[Path]] = {}
    for path in sorted(directory.glob("*.whl")):
        name, wheel_version, _, _ = parse_wheel_filename(path.name)
        if name in INTERNAL_DISTRIBUTIONS:
            if wheel_version != Version(version):
                raise ValueError(f"{path.name} is not candidate version {version}")
            wheels.setdefault(name, []).append(path.resolve())
    for name in names:
        if len(wheels.get(name, [])) != 1:
            raise ValueError(
                f"Expected one candidate wheel for {name}; no PyPI fallback permitted"
            )
    return [wheels[name][0] for name in names]


def build_wheels(root: Path, version: str, output: Path) -> None:
    candidate_version(root, version)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Wheel output must be empty to avoid stale artifacts: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    environment = clean_environment()
    environment["SETUPTOOLS_SCM_PRETEND_VERSION"] = version
    for package in PACKAGES.values():
        run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(output),
                str(root / package["path"]),
            ],
            root,
            environment,
        )
    wheel_artifacts(list(PACKAGE_ORDER), version, output)


def release_available(root: Path, version: str) -> bool:
    status = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/tags/v{version}"],
        cwd=root,
        check=False,
    ).returncode
    if status not in {0, 1}:
        raise ValueError(f"Cannot inspect release tag v{version}: git exited {status}")
    return status == 0


# Executed by the validation interpreter, which deliberately has no tooling installed.
VERIFY_INSTALLATION = """
import json
import sys
import sysconfig
from importlib import metadata, util
from pathlib import Path

expected = json.loads(sys.argv[1])
internal = set(json.loads(sys.argv[2]))
strict = sys.argv[3] == "strict"
installed = {d.metadata["Name"].lower().replace("_", "-").replace(".", "-")
             for d in metadata.distributions()}
if strict and installed & internal != set(expected):
    raise SystemExit(f"Unexpected internal distributions: {installed & internal}; expected {set(expected)}")
sites = {Path(sysconfig.get_path(name)).resolve() for name in ("purelib", "platlib")}
for name, version in expected.items():
    actual = metadata.version(name)
    if actual != version:
        raise SystemExit(f"{name} is {actual}, expected {version}")
for name in installed & internal:
    module = util.find_spec(name.replace("-", "_"))
    if module is None or module.origin is None or not any(
        Path(module.origin).resolve().is_relative_to(site) for site in sites
    ):
        raise SystemExit(f"{name} did not load from the isolated wheel environment")
print("Verified installed versions and wheel import origins")
"""


def verify_installation(
    root: Path,
    python: Path,
    expected: dict[str, str],
    *,
    strict: bool,
    environment: dict[str, str],
) -> None:
    run(
        [
            str(python),
            "-c",
            VERIFY_INSTALLATION,
            json.dumps(expected),
            json.dumps(list(PACKAGE_ORDER)),
            "strict" if strict else "minimum",
        ],
        root,
        environment,
    )


def validate(
    root: Path,
    distribution: str,
    version: str,
    wheels: Path,
    *,
    minimum: bool = False,
) -> MinimumStatus:
    candidate_version(root, version)
    if minimum:
        expected = minimum_bounds(root, distribution)
        if not expected:
            print(f"{distribution}: minimum not-applicable (no internal dependencies)")
            return "not-applicable"
        missing: set[str] = set()
        for bound in set(expected.values()):
            if not release_available(root, bound):
                missing.add(bound)
        if missing:
            tags = ", ".join(f"v{v}" for v in sorted(missing))
            print(f"{distribution}: minimum unreleased / not verified: {tags}")
            return "unreleased"
        print(f"{distribution}: testing published minimum dependencies")
        artifacts = wheel_artifacts([distribution], version, wheels)
    else:
        expected = {name: version for name in candidate_closure(root, distribution)}
        artifacts = wheel_artifacts(list(expected), version, wheels)
    expected[distribution] = version
    environment = clean_environment()
    with tempfile.TemporaryDirectory(prefix="sefia-compat-") as temporary:
        directory = Path(temporary)
        run([sys.executable, "-m", "venv", str(directory / "venv")], root, environment)
        python = (
            directory
            / "venv"
            / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
        constraints = directory / "constraints.txt"
        pins = expected if minimum else {name: version for name in PACKAGE_ORDER}
        constraints.write_text("".join(f"{name}=={v}\n" for name, v in pins.items()))
        pip = [str(python), "-m", "pip", "install", "--constraint", str(constraints)]
        target = PACKAGES[distribution]["requirement"]
        if minimum:
            requirement = f"{target} @ {artifacts[0].as_uri()}"
            run([*pip, requirement, *TEST_REQUIREMENTS], root, environment)
        else:
            run(
                [*pip, "--no-index", "--no-deps", *(str(path) for path in artifacts)],
                root,
                environment,
            )
            run([*pip, f"{target}=={version}", *TEST_REQUIREMENTS], root, environment)
        verify_installation(
            root, python, expected, strict=not minimum, environment=environment
        )
        run([str(python), "-m", "pip", "check"], root, environment)
        run(
            [
                str(python),
                "-m",
                "pytest",
                str(root / PACKAGES[distribution]["tests"]),
                "--import-mode=importlib",
                "--asyncio-mode=auto",
                "-m",
                "not e2e",
            ],
            root,
            environment,
        )
    print(f"{distribution}: {'minimum' if minimum else 'candidate'} verified")
    return "available"
