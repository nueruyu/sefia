# sefia-typer

Typer (CLI) building blocks for [Sefia](https://pypi.org/project/sefia/)
applications.

This package holds the CLI-side pieces that depend only on `sefia` and Typer:
the framework-neutral `sefia.input_channels.InputChannel` with CLI reporting hooks,
the reporter surface, and the exceptions applications catch. The runtime wiring —
session management, persistence, and the pausing tool — is provided by an
integration layer such as `sefios.cli` from
[`sefios`](https://pypi.org/project/sefios/).

Import exception types from `sefia_typer.exceptions`; they are intentionally not
re-exported from the package root.

## Install

```bash
pip install sefia-typer
```

Most applications install it through the stack instead:

```bash
pip install 'sefios[cli]'
```

## Documentation

See the [repository](https://github.com/nueruyu/sefia) for the full README,
tutorial, and architecture docs.

## Status

Early development. APIs may change before v1.0.

## License

MIT
