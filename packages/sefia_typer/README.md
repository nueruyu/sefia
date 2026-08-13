# sefia-typer

Typer (CLI) building blocks for [Sefia](https://pypi.org/project/sefia/)
applications.

This package holds CLI reporter types and exceptions that depend only on `sefia`
and Typer. Runtime wiring, input routing, persistence, and pausing tools are
provided by an integration layer such as `sefios.cli` from
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
