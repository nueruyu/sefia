"""Typed adapters for Typer's public parameter factories."""

from collections.abc import Callable
from typing import cast

import typer

_argument_factory = cast(Callable[..., object], getattr(typer, "Argument"))
_option_factory = cast(Callable[..., object], getattr(typer, "Option"))


def argument(*, help: str) -> object:
    return _argument_factory(help=help)


def option(
    *param_decls: str,
    envvar: str | list[str] | None = None,
    help: str | None = None,
) -> object:
    return _option_factory(*param_decls, envvar=envvar, help=help)
