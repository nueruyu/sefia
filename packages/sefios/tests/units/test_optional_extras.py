"""The optional integration packages fail with an actionable install hint."""

import builtins
import importlib
import importlib.util as importlib_util
import sys
from importlib.machinery import ModuleSpec
from typing import Any

import pytest
from sefios import SQLitePersistence


def _unload_package(monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    owner, module_name = module.rsplit(".", 1)
    package = importlib.import_module(owner)
    monkeypatch.delattr(package, module_name, raising=False)
    for loaded in tuple(sys.modules):
        if loaded == module or loaded.startswith(f"{module}."):
            monkeypatch.delitem(sys.modules, loaded)


@pytest.mark.parametrize(
    ("module", "adapter", "extra"),
    [
        ("sefios.cli", "sefia_typer", "cli"),
        ("sefios.fastapi", "sefia_fastapi", "fastapi"),
    ],
)
def test_missing_adapter_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    adapter: str,
    extra: str,
) -> None:
    _unload_package(monkeypatch, module)

    real_find_spec = importlib_util.find_spec
    real_import = builtins.__import__

    def fake_find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name == adapter:
            return None
        return real_find_spec(name, package)

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if (name == adapter or name.startswith(f"{adapter}.")) and fromlist:
            raise ModuleNotFoundError(f"No module named '{adapter}'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=rf"pip install 'sefios\[{extra}\]'"):
        importlib.import_module(module)


@pytest.mark.parametrize(
    ("module", "adapter"),
    [
        ("sefios.cli", "sefia_typer"),
        ("sefios.fastapi", "sefia_fastapi"),
    ],
)
def test_adapter_import_errors_are_not_reported_as_missing_extra(
    monkeypatch: pytest.MonkeyPatch, module: str, adapter: str
) -> None:
    _unload_package(monkeypatch, module)

    real_find_spec = importlib_util.find_spec
    real_import = builtins.__import__

    def fake_find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name == adapter:
            return ModuleSpec(name, loader=None)
        return real_find_spec(name, package)

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if (name == adapter or name.startswith(f"{adapter}.")) and fromlist:
            raise ImportError("adapter dependency exploded")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="adapter dependency exploded"):
        importlib.import_module(module)


def test_sqlite_persistence_requires_the_sqlite_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "glyff_sqlite":
            raise ModuleNotFoundError("No module named 'glyff_sqlite'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"pip install 'sefios\[sqlite\]'"):
        SQLitePersistence("sessions.sqlite3")
