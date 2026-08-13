"""The optional integration packages fail with an actionable install hint."""

import builtins
import importlib
import importlib.util as importlib_util
import sys
from importlib.machinery import ModuleSpec

import pytest


def _unload_package(monkeypatch, module):
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
def test_missing_adapter_raises_install_hint(monkeypatch, module, adapter, extra):
    _unload_package(monkeypatch, module)

    real_find_spec = importlib_util.find_spec
    real_import = builtins.__import__

    def fake_find_spec(name, *args, **kwargs):
        if name == adapter:
            return None
        return real_find_spec(name, *args, **kwargs)

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
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
    monkeypatch, module, adapter
):
    _unload_package(monkeypatch, module)

    real_find_spec = importlib_util.find_spec
    real_import = builtins.__import__

    def fake_find_spec(name, *args, **kwargs):
        if name == adapter:
            return ModuleSpec(name, loader=None)
        return real_find_spec(name, *args, **kwargs)

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (name == adapter or name.startswith(f"{adapter}.")) and fromlist:
            raise ImportError("adapter dependency exploded")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="adapter dependency exploded"):
        importlib.import_module(module)
