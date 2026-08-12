# sefia-fastapi

FastAPI (HTTP) building blocks for [Sefia](https://pypi.org/project/sefia/)
applications.

This package holds the HTTP-side pieces that depend only on `sefia` and
FastAPI: the framework-neutral `sefia.input_channels.InputChannel` persisted over a
`KeyValueStore`,
per-session SSE streams (`sefia_fastapi.events`), and the exceptions an application
maps to HTTP responses (`sefia_fastapi.exceptions`). The runtime wiring — session management, persistence,
and the pausing tool — is provided by an integration layer such as
`sefios.fastapi` from [`sefios`](https://pypi.org/project/sefios/).

## Install

```bash
pip install sefia-fastapi
```

Most applications install it through the stack instead:

```bash
pip install 'sefios[fastapi]'
```

## Documentation

See the [repository](https://github.com/nueruyu/sefia) for the full README,
tutorial, and architecture docs.

## Status

Early development. APIs may change before v1.0.

## License

MIT
