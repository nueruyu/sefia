import json
from pathlib import Path

import pytest
from package_compatibility import dependencies


@pytest.fixture
def manifests(tmp_path: Path) -> Path:
    dependency = "sefia>=0.6.0,<0.7.0"
    extras = {
        "sefia": {"testing": ["pytest>=8.0"]},
        "sefios": {
            "litellm": ["sefia-litellm>=0.6.0,<0.7.0"],
            "cli": ["sefia-typer>=0.6.0,<0.7.0"],
            "fastapi": ["sefia-fastapi>=0.6.0,<0.7.0"],
            "all": ["sefios[litellm]", "sefios[cli]", "sefios[fastapi]"],
        },
    }
    for name, package in dependencies.PACKAGES.items():
        destination = tmp_path / package["path"] / "pyproject.toml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        requirements = [] if name == "sefia" else [dependency]
        lines = [
            "[project]",
            f"name = {json.dumps(name)}",
            f"dependencies = {json.dumps(requirements)}",
        ]
        if name in extras:
            lines.append("[project.optional-dependencies]")
            lines.extend(
                f"{extra} = {json.dumps(requirements)}"
                for extra, requirements in extras[name].items()
            )
        destination.write_text("\n".join(lines) + "\n")
    (tmp_path / "pyproject.toml").write_text("[tool.uv.workspace]\nmembers = []\n")
    return tmp_path
