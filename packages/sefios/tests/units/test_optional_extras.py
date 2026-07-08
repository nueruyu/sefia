"""The optional integration packages fail with an actionable install hint."""

import builtins
import importlib
import sys

import pytest


@pytest.mark.parametrize(
    ("module", "adapter", "extra"),
    [
        ("sefios.cli", "sefia_typer", "cli"),
        ("sefios.fastapi", "sefia_fastapi", "fastapi"),
    ],
)
def test_missing_adapter_raises_install_hint(monkeypatch, module, adapter, extra):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == adapter:
            raise ImportError(f"No module named '{adapter}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, module, raising=False)

    with pytest.raises(ImportError, match=rf"pip install 'sefios\[{extra}\]'"):
        importlib.import_module(module)
