from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import dependencies, validation


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Validate Sefia package wheels outside the workspace."
    )
    commands = result.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--version")
    build = commands.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--version", required=True)
    for command in ("validate-candidate", "validate-minimum"):
        check = commands.add_parser(command)
        check.add_argument(
            "--distribution", choices=dependencies.PACKAGE_ORDER, required=True
        )
        check.add_argument("--version", required=True)
        check.add_argument("--wheels", type=Path, required=True)
    for command in commands.choices.values():
        command.add_argument("--root", type=Path, default=Path.cwd())
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        root = dependencies.repository_root(args.root)
        if args.command == "plan":
            for name in dependencies.PACKAGE_ORDER:
                dependencies.candidate_closure(root, name)
            version = dependencies.candidate_version(root, args.version)
            source = (
                "explicit intended release" if args.version else "derived PR candidate"
            )
            print(f"Candidate version: {version} ({source})", file=sys.stderr)
            plan = {
                "candidate_version": version,
                "matrix": {
                    "include": [
                        {"distribution": name} for name in dependencies.PACKAGE_ORDER
                    ]
                },
            }
            print(json.dumps(plan, separators=(",", ":")))
        elif args.command == "build":
            validation.build_wheels(root, args.version, args.output)
        else:
            status = validation.validate(
                root,
                args.distribution,
                args.version,
                args.wheels,
                minimum=args.command == "validate-minimum",
            )
            return {"available": 0, "unreleased": 10, "not-applicable": 11}[status]
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(
            f"Package compatibility failed ({args.command}): {error}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
