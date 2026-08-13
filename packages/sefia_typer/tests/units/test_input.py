import sefia_typer
from sefia.input_channels import InputChannel, InputRequest, KeyValueStore
from sefia_typer import exceptions


def test_input_core_is_reexported_from_package_root():
    assert sefia_typer.InputChannel is InputChannel
    assert sefia_typer.InputRequest is InputRequest
    assert sefia_typer.KeyValueStore is KeyValueStore


def test_input_exceptions_are_only_exposed_by_the_exceptions_module():
    assert exceptions.AmbiguousInputError.__module__ == "sefia.input_channels"
    assert exceptions.UnknownInputError.__module__ == "sefia.input_channels"
    assert "AmbiguousInputError" not in sefia_typer.__all__
    assert "UnknownInputError" not in sefia_typer.__all__
    assert not hasattr(sefia_typer, "AmbiguousInputError")
    assert not hasattr(sefia_typer, "UnknownInputError")
